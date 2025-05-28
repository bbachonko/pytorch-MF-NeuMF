from collections import defaultdict
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from recommender.utils.logger import setup_logger
from recommender.engine.losses import bpr_loss
from recommender.models.mf import MatrixFactorization
from recommender.models.neumf import NeuMFHybrid

logger = setup_logger(__name__)


def train_mf(
    interactions_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    *,
    embedding_dim: int = 64,
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 5e-4,
    negatives_per_pos: int = 1,
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MatrixFactorization:
    """
    Train a Matrix-Factorisation model with BPR loss and **dynamic negative
    sampling**.  Uses AdamW (dense gradients) because embeddings are stored
    as dense tensors – SparseAdam is not legal here.
    """
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    model = MatrixFactorization(num_users, num_items, embedding_dim).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    # Pre-filter positives once; we resample negatives every epoch
    positives_df = interactions_df[interactions_df.label == 1]
    all_item_ids = np.arange(num_items)

    for epoch_idx in range(1, epochs + 1):
        # ── 1. Resample negatives – avoids memorising a fixed set ──────────
        triplets: list[tuple[int, int, int]] = []   # (user, pos, neg)
        for user_id, group in positives_df.groupby("user"):
            positive_items = group.item.values
            candidate_items = np.setdiff1d(all_item_ids, positive_items, assume_unique=True)
            if candidate_items.size == 0:
                continue  # user interacted with *all* items
            negative_items = rng.choice(
                candidate_items,
                size=len(positive_items) * negatives_per_pos,
                replace=False,
            )
            # Tile positives so we can zip() with negatives 1-to-1
            tiled_pos = np.tile(positive_items, negatives_per_pos)
            triplets.extend([(user_id, p, n) for p, n in zip(tiled_pos, negative_items)])

        # ── 2. DataLoader with pinned memory for async CPU→GPU copies ─────
        if not triplets:
            logger.warning(f"No training samples generated in epoch {epoch_idx}.")
            continue

        users_np, pos_np, neg_np = map(np.asarray, zip(*triplets))
        dataset = TensorDataset(
            torch.as_tensor(users_np, dtype=torch.long),
            torch.as_tensor(pos_np,   dtype=torch.long),
            torch.as_tensor(neg_np,   dtype=torch.long),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )

        # ── 3. Mini-batch optimisation ────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for user_batch, pos_batch, neg_batch in loader:
            user_batch = user_batch.to(device, non_blocking=True)
            pos_batch  = pos_batch.to(device,  non_blocking=True)
            neg_batch  = neg_batch.to(device,  non_blocking=True)

            pos_scores = model(user_batch, pos_batch)
            neg_scores = model(user_batch, neg_batch)
            loss = bpr_loss(pos_scores, neg_scores)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        mean_bpr = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch_idx}/{epochs} | mean BPR loss = {mean_bpr:.4f}")

    return model


def train_neumf_hybrid(
    train_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    content_matrix: np.ndarray,
    *,
    embedding_dim: int = 32,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 5e-4,
    seed: int = 42,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> "NeuMFHybrid":
    """Train *NeuMFHybrid* under **pairwise BPR**.

    The model receives content vectors for **both** positive and negative items
    so it can differentiate via metadata even when IDs are unseen.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    content_dim = content_matrix.shape[1]
    model = NeuMFHybrid(
        num_users,
        num_items,
        content_dim,
        emb_dim=embedding_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- positive ↔ negative triplets ---------------------------------

    pos_items_by_user: dict[int, list[int]] = defaultdict(list)
    for row in train_df.itertuples():
        if row.label == 1:
            pos_items_by_user[row.user].append(row.item)

    all_items = np.arange(num_items)
    triplets: list[tuple[int, int, int]] = []
    for user, pos_list in pos_items_by_user.items():
        pos_set = set(pos_list)
        neg_candidates = np.setdiff1d(all_items, list(pos_set), assume_unique=True)
        for pos_item in pos_list:
            if len(neg_candidates) == 0:
                continue
            neg_item = rng.choice(neg_candidates)
            triplets.append((user, pos_item, neg_item))

    users_t = torch.as_tensor([u for u, _, _ in triplets], dtype=torch.long, device=device)
    pos_t = torch.as_tensor([p for _, p, _ in triplets], dtype=torch.long, device=device)
    neg_t = torch.as_tensor([n for _, _, n in triplets], dtype=torch.long, device=device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        indices = rng.permutation(len(users_t))
        content_matrix_torch = torch.as_tensor(content_matrix, device=device)
        for start in range(0, len(indices), batch_size):
            # Slicing current epoch specific batch
            batch_idx = indices[start : start + batch_size]
            u = users_t[batch_idx]
            i_pos = pos_t[batch_idx]
            i_neg = neg_t[batch_idx]

            # Directly slice from preloaded GPU tensor
            c_pos = content_matrix_torch[i_pos]
            c_neg = content_matrix_torch[i_neg]

            pos_scores = model(u, i_pos, c_pos)
            neg_scores = model(u, i_neg, c_neg)
            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}/{epochs} | mean BPR loss = {epoch_loss / (len(users_t) // batch_size):.4f}")

    return model