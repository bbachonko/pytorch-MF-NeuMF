import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from recommender.utils.logger import setup_logger
from recommender.data.preprocess import MFPreprocessor, leave_one_out_split, sample_negatives
from recommender.engine.metrics import evaluate_topk
from recommender.engine.trainer import train_mf


logger = setup_logger(__name__)


def run_pipeline(
    ratings: pd.DataFrame,
    *,
    min_ratings: int = 5,
    threshold: int = 4,
    negatives_per_pos: int = 4,
    eval_K: int = 10,
    eval_n_neg: int = 99,
    seed: int = 42,
    **trainer_kwargs,
):
    """End‑to‑end run: preprocess → split → sample negatives → train → evaluate.

    Parameters
    ----------
    ratings : pd.DataFrame
        DataFrame containing columns 'user_id', 'book_id', and 'rating'.
    min_ratings : int, optional
        Minimum number of interactions required for users and items during filtering,
        by default 5.
    threshold : int, optional
        Rating threshold for positive interactions (>= threshold), by default 4.
    negatives_per_pos : int, optional
        Number of negative samples to draw per positive example, by default 4. Used in training process.
    eval_K : int, optional
        Cutoff rank for computing HR@K, nDCG@K, and MRR@K, by default 10. Mst be less than `eval_n_neg`
        since its actually a number of subset for `eval_n_neg`
    eval_n_neg : int, optional
        Number of negative candidates sampled per user during evaluation,
        by default 99. Used in evalution process.
    seed : int, optional
        Random seed for all sampling and splits to ensure reproducibility,
        by default 42.
    **trainer_kwargs
        Additional keyword arguments passed to the training function,
        such as 'epochs', 'embedding_dim', 'batch_size', and 'lr'.

    Returns
    -------
    model : MatrixFactorization
        Trained MF model instance.
    metrics : dict[str, float]
        Evaluation metrics: keys include 'HR@K', 'nDCG@K', and 'MRR@K'.
    data_objects : tuple[pd.DataFrame, pd.DataFrame, MFPreprocessor]
        A tuple containing (train_df, test_df, preprocessor) for further analysis.

    """
    prep = MFPreprocessor(ratings, min_ratings=min_ratings, threshold=threshold, seed=seed)
    pos_df = prep.process()

    train_pos, test_pos = leave_one_out_split(pos_df, seed=seed)
    neg_train = sample_negatives(train_pos, prep.num_items(), negatives_per_pos=negatives_per_pos, seed=seed)
    train_df = (
        pd.concat([train_pos, neg_train])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    logger.info(
        f"Training samples: {len(train_df):,}  |  positives: {len(train_pos):,}  |  negatives: {len(neg_train):,}"
    )

    model = train_mf(
        train_df,
        num_users=prep.num_users(),
        num_items=prep.num_items(),
        **trainer_kwargs,
    )

    metrics_train = evaluate_topk(
        model,
        train_pos,
        train_pos,
        prep.num_items(),   # test_df_pos == train_pos
        K=eval_K,
        n_neg=eval_n_neg,
        seed=seed,
    )
    metrics_test = evaluate_topk(
        model,
        test_pos,
        train_pos,
        prep.num_items(),
        K=eval_K,
        n_neg=eval_n_neg,
        seed=seed,
    )

    logger.info("\nTrain set Evaluation:")
    for k, v in metrics_train.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("\nTest set Evaluation:")
    for k, v in metrics_test.items():
        logger.info(f"  {k}: {v:.4f}")

    return model, metrics_test, (train_df, test_pos, prep)
