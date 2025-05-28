import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuMFHybrid(nn.Module):
    """
    NeuMFHybrid
    ===========

    A hybrid version of *Neural Matrix-Factorization* (He et al., 2017) that
    mixes classic collaborative signals (user-ID × item-ID) with arbitrary
    **item-side content features**—tags, author, year, language, popularity
    stats, …—all fed through a single linear *projection* layer.

    Branches
    --------
    • **GMF branch** – learns a *linear* interaction between user and item
      IDs, i.e. the classic element-wise product of two embeddings.

    • **MLP branch** – learns *non-linear* interactions between:

        ┌ user embedding (MLP table)
        │
        ├ item embedding (MLP table)
        │
        └ projected content vector  ←  Linear(content_dim → emb_dim)

      The three pieces are concatenated, passed through a small MLP, and
      then fused with the GMF signal for the final prediction.
    """

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        n_users: int,               # number of unique users
        n_items: int,               # number of unique items
        content_dim: int,           # raw content feature length
        emb_dim: int = 32,          # embedding size for IDs *and* content
        mlp_layers: tuple[int, ...] = (64, 32),  # hidden widths after concat
    ):
        super().__init__()

        # --- 1.  ID embedding tables ---------------------------------
        # GMF (Generalised MF) branch → purely linear ID × ID signal
        self.user_gmf = nn.Embedding(n_users, emb_dim)
        self.item_gmf = nn.Embedding(n_items, emb_dim)

        # MLP branch uses *separate* tables so it can learn a different
        # representation from GMF.
        self.user_mlp = nn.Embedding(n_users, emb_dim)
        self.item_mlp = nn.Embedding(n_items, emb_dim)

        # --- 2.  Content projection ----------------------------------
        # Any high-dimensional content vector is squashed to `emb_dim`
        # so the subsequent MLP sees a *fixed* width no matter how many
        # features you add later.
        self.content_proj = nn.Linear(content_dim, emb_dim, bias=False)

        # --- 3.  MLP tower -------------------------------------------
        # Input: [user_mlp | item_mlp | projected_content] of size 3·emb_dim
        mlp_in = emb_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_layers[0]),
            nn.ReLU(),
            nn.Linear(mlp_layers[0], mlp_layers[1]),
            nn.ReLU(),
        )

        # --- 4.  Output layer ----------------------------------------
        # Concatenate GMF vector (emb_dim) with last MLP output, then a
        # single linear unit predicts a logit; we apply sigmoid in forward.
        self.out = nn.Linear(emb_dim + mlp_layers[-1], 1, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # custom weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self):
        # Small N(0, 0.01) for all embeddings
        for emb in (
            self.user_gmf,
            self.item_gmf,
            self.user_mlp,
            self.item_mlp,
        ):
            nn.init.normal_(emb.weight, std=0.01)

        # Xavier-uniform for every Linear layer
        for mod in (self.content_proj, *self.mlp, self.out):
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)

    def forward(
        self,
        users: torch.Tensor,          # LongTensor  shape (B,)
        items: torch.Tensor,          # LongTensor  shape (B,)
        content_vec: torch.Tensor,    # FloatTensor shape (B, content_dim)
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        users, items : torch.LongTensor
            Indices of the user and the item in the batch.
        content_vec : torch.FloatTensor
            Pre-computed content feature vector for each *item* in the batch
            (already aligned by `items` index order).

        Returns
        -------
        probs : torch.FloatTensor shape (B,)
            Interaction probability in (0, 1).
        """

        # ---------- GMF branch ---------- #
        # Purely collaborative: element-wise product of ID embeddings
        gmf_u = self.user_gmf(users)           # (B, emb_dim)
        gmf_i = self.item_gmf(items)           # (B, emb_dim)
        gmf_vec = gmf_u * gmf_i                # (B, emb_dim)

        # ---------- MLP branch ---------- #
        # 1) ID embeddings
        mlp_u = self.user_mlp(users)           # (B, emb_dim)
        mlp_i = self.item_mlp(items)           # (B, emb_dim)

        # 2) Project content to emb_dim and add non-linearity
        proj = F.relu(self.content_proj(content_vec))  # (B, emb_dim)

        # 3) Concatenate and feed through the two-layer MLP
        mlp_input = torch.cat([mlp_u, mlp_i, proj], dim=1)   # (B, 3·emb_dim)
        mlp_out = self.mlp(mlp_input)                        # (B, mlp_layers[-1])

        # ---------- Fusion + prediction ---------- #
        fusion = torch.cat([gmf_vec, mlp_out], dim=1)        # (B, emb_dim+mlp_last)
        logits = self.out(fusion).squeeze(-1)                # (B,)

        return torch.sigmoid(logits)                         # p-lities
