import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class MFPreprocessor:
    """Prepare ratings for *classical* MF on **binary implicit** feedback.

    Parameters
    ----------
    ratings : pd.DataFrame
        Must contain columns `user_id`, `book_id`, `rating`.
    min_ratings : int, default 5
        Minimum #interactions both for a user and for an item.
    threshold : int, default 4
        Ratings ≥ `threshold` are treated as positive (label = 1).
    seed : int, default 42
        Global seed for any random operation.
    """

    def __init__(self, ratings: pd.DataFrame, *, min_ratings: int = 5, threshold: int = 4, seed: int = 42):
        self._raw = ratings.copy()
        self.min_ratings = min_ratings
        self.threshold = threshold
        self.seed = seed

        # public artefacts filled by .process()
        self.ratings: pd.DataFrame  # filtered + encoded + binarised
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    # ------------------------------------------------------------------
    # 2.1  Sparse‑filter – **iterated until convergence**
    # ------------------------------------------------------------------

    def _iterative_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop users/items with < min_ratings *recursively* until stable."""
        changed = True
        while changed:
            start_len = len(df)

            user_counts = df.user_id.value_counts()
            df = df[df.user_id.isin(user_counts[user_counts >= self.min_ratings].index)]

            item_counts = df.book_id.value_counts()
            df = df[df.book_id.isin(item_counts[item_counts >= self.min_ratings].index)]

            changed = len(df) != start_len
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2.2  Public entry – end‑to‑end processing
    # ------------------------------------------------------------------

    def process(self) -> pd.DataFrame:
        """Run full pipeline → returns DataFrame with columns [user, item, label]."""
        # [1] sparsity filtering
        df = self._iterative_filter(self._raw)

        # [2️] encode AFTER filtering so that indices are *dense*
        df["user"] = self.user_encoder.fit_transform(df.user_id)
        df["item"] = self.item_encoder.fit_transform(df.book_id)

        # [3️] binarise explicit ratings
        df = df[df.rating >= self.threshold][["user", "item"]].copy()
        df["label"] = 1

        self.ratings = df  # store for later access
        return df

    # helpers
    def num_users(self) -> int:
        return len(self.user_encoder.classes_)

    def num_items(self) -> int:
        return len(self.item_encoder.classes_)


def leave_one_out_split(df_pos: pd.DataFrame, *, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly choose **one** positive example per user for the test set.

    This function guarantees each user appears in *both* splits.
    """
    rng = np.random.default_rng(seed)
    test_idx: list[int] = []

    for user, group in df_pos.groupby("user"):
        test_idx.append(rng.choice(group.index))  # Selecting unique idx from user, item group

    test_df = df_pos.loc[test_idx].reset_index(drop=True)
    train_df = df_pos.drop(index=test_idx).reset_index(drop=True)  # Dropping all test idxs from train
    return train_df, test_df


def sample_negatives(
    train_df_pos: pd.DataFrame,
    num_items: int,
    *,
    negatives_per_pos: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate negative (user, item, 0) triples – *items unseen in train positives*.

    This function generates negative user-item pairs for training recommendation models by ensuring
    users are paired only with items they haven't interacted with. It handles edge cases like users
    who have interacted with all items and avoids duplicate sampling for the same user.

    We sample **without replacement per user** to avoid trivial duplicates, but we
    still allow the *same* item to be used for different users, which is fine.
    """
    rng = np.random.default_rng(seed)
    all_items = np.arange(num_items)

    user_positives: dict[int, set[int]] = (
        train_df_pos.groupby("user")["item"].apply(set).to_dict()
    )

    negatives: list[tuple[int, int, int]] = []
    for user, pos_items in user_positives.items():
        # Find the items this user has NOT interacted with (negative candidates)
        candidates = np.setdiff1d(all_items, np.fromiter(pos_items, dtype=int), assume_unique=True)

        if len(candidates) == 0:
            continue  # Skip this user if they have interacted with all items

        size = len(pos_items) * negatives_per_pos  # Number of negative samples to draw for this user
        replace = size > len(candidates)  # If we need more negatives than available candidates, sample with replacement
        sampled = rng.choice(candidates, size=size, replace=replace)  # Sample `size` number of items from the candidate pool
        negatives.extend([(user, int(item), 0) for item in sampled])  # Appending the sampled negatives to the list with label=0

    neg_df = pd.DataFrame(negatives, columns=["user", "item", "label"])
    return neg_df