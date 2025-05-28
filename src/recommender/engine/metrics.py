from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from recommender.models.neumf import NeuMFHybrid
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from recommender.utils import logger
from recommender.models.mf import MatrixFactorization


def _rank_metrics(rank: int, K: int) -> tuple[int, float, float]:
    """Return (hit, ndcg, mrr) for a single user given *rank* of the positive."""
    hit = 1 if rank < K else 0  # higher when more relevant items appers in top K reccs.
    ndcg = (1 / np.log2(rank + 2)) if hit else 0.0  # Gives higher score to correct items that appear earlier in the ranking.
    mrr = 1 / (rank + 1) if hit else 0.0  # How early the first relevant item appears
    return hit, ndcg, mrr


def evaluate_topk(
    model: MatrixFactorization,
    test_df_pos: pd.DataFrame,
    train_df_pos: pd.DataFrame,
    num_items: int,
    *,
    K: int = 10,
    n_neg: int = 99,
    seed: int = 42,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, float]:
    """Compute HR@K, nDCG@K, MRR@K on leave‑one‑out test.

    Each user is evaluated on 1 positive + *n_neg* sampled negatives
    never seen in **training**. This mirrors common practice in MF papers
    (He et al., 2017 *NeuMF*).
    99 negavtives per 1 positive is also common practice.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    model.to(device)

    all_items = np.arange(num_items)
    hits, ndcgs, mrrs = [], [], []

    # pre‑compute training items per user to speed up masking
    train_items_by_user: dict[int, set[int]] = (
        train_df_pos.groupby("user")["item"].apply(set).to_dict()
    )

    with torch.no_grad():
        for user, pos_item in tqdm(
            test_df_pos[["user", "item"]].itertuples(index=False, name=None),
            total=len(test_df_pos),
            desc="Evaluating",
            unit="user",
            bar_format="{l_bar}{bar:30} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            user = int(user)
            pos_item = int(pos_item)

            forbidden = train_items_by_user.get(user, set()).union({pos_item})
            candidates = np.setdiff1d(all_items, np.fromiter(forbidden, dtype=int), assume_unique=True)

            if len(candidates) == 0:
                continue  # extremely dense user – skip

            neg_items = rng.choice(candidates, size=min(n_neg, len(candidates)), replace=False)
            item_ids = np.concatenate(([pos_item], neg_items))
            user_ids = np.full_like(item_ids, user)

            users_t = torch.as_tensor(user_ids, dtype=torch.long, device=device)
            items_t = torch.as_tensor(item_ids, dtype=torch.long, device=device)
            scores = model(users_t, items_t).cpu().numpy()

            # rank descending by score
            rank = int(np.where(np.argsort(-scores) == 0)[0])
            hit, ndcg, mrr = _rank_metrics(rank, K)
            hits.append(hit)
            ndcgs.append(ndcg)
            mrrs.append(mrr)

    return {
        f"HR@{K}": float(np.mean(hits)),
        f"nDCG@{K}": float(np.mean(ndcgs)),
        f"MRR@{K}": float(np.mean(mrrs)),
    }


def align_content_matrix(content_df: pd.DataFrame, item_encoder: LabelEncoder) -> np.ndarray:
    """Re‑order rows so that *row i* corresponds to *item i* in the model.

    This is **crucial** – a single off‑by‑one error here silently destroys
    training.
    """
    aligned = content_df.reindex(item_encoder.classes_).fillna(0.0)
    assert not aligned.isnull().values.any(), "Content matrix contains NaNs after re‑indexing."
    return aligned.values.astype(np.float32)


def evaluate_topk_hybrid(
    model: NeuMFHybrid,
    test_df_pos: "pd.DataFrame",
    train_df_pos: "pd.DataFrame",
    num_items: int,
    content_matrix: np.ndarray,
    *,
    k: int = 10,
    n_neg: int = 99,
    batch_users: int = 1024,
    seed: int = 42,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, float]:
    """
    Vectorised HR/nDCG/MRR evaluation that is **O(#users + #items) GPU passes**
    instead of O(#users).
    """
    rng   = np.random.default_rng(seed)
    model = model.to(device).eval()

    # ── pre-compute helpers ────────────────────────────────────────────────
    all_items = np.arange(num_items)
    train_pos = train_df_pos.groupby("user")["item"].apply(set).to_dict()
    content_t = torch.as_tensor(content_matrix, device=device)

    hits, ndcgs, mrrs = [], [], []

    # ── iterate users in mini-batches ──────────────────────────────────────
    uid_batch: list[int] = []
    iid_batch: list[int] = []
    len_batch: list[int] = []          # (1 + #neg) for every user in uid_batch

    def _flush():
        """Run the torch model once for the current batch and accumulate metrics."""
        if not uid_batch:
            return

        users_t   = torch.as_tensor(uid_batch, dtype=torch.long, device=device)
        items_t   = torch.as_tensor(iid_batch, dtype=torch.long, device=device)
        content_t_batch = content_t[items_t]

        # forward pass in one go
        scores = model(users_t, items_t, content_t_batch)  # flat tensor
        start = 0
        for ln in len_batch:
            row = scores[start : start + ln].cpu().numpy()
            start += ln
            # rank = int(np.where(np.argsort(-row) == 0)[0])
            rank = int(np.where(np.argsort(-row) == 0)[0][0])
            hit, ndcg, mrr = _rank_metrics(rank, k)
            hits.append(hit); ndcgs.append(ndcg); mrrs.append(mrr)

        # clear buffers
        uid_batch.clear(); iid_batch.clear(); len_batch.clear()

    with torch.no_grad():
        for user, pos_item in tqdm(
            test_df_pos[["user", "item"]].itertuples(index=False, name=None),
            total=len(test_df_pos),
            desc="Evaluating",
            unit="user",
            bar_format="{l_bar}{bar:30} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            user = int(user)
            forbidden = train_pos.get(user, set()).union({pos_item})
            candidates = np.setdiff1d(all_items,
                                      np.fromiter(forbidden, dtype=int),
                                      assume_unique=True)
            if len(candidates) == 0:
                continue

            neg_items = rng.choice(candidates,
                                   size=min(n_neg, len(candidates)),
                                   replace=False)

            item_ids = np.concatenate(([pos_item], neg_items))

            # fill buffers
            uid_batch.extend([user] * len(item_ids))
            iid_batch.extend(item_ids)
            len_batch.append(len(item_ids))

            # flush if we reached batch size
            if len(len_batch) >= batch_users:
                _flush()

        _flush()   # leftover users

    return {f"HR@{k}": float(np.mean(hits)),
            f"nDCG@{k}": float(np.mean(ndcgs)),
            f"MRR@{k}": float(np.mean(mrrs))}
