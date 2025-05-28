from typing import Dict
import pandas as pd

from recommender.data.features import build_item_content_matrix
from recommender.data.preprocess import MFPreprocessor, leave_one_out_split, sample_negatives
from recommender.engine.metrics import align_content_matrix, evaluate_topk_hybrid
from recommender.engine.trainer import train_neumf_hybrid
from recommender.utils.logger import  setup_logger

logger = setup_logger(__name__)

def run_pipeline_neumf(
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    # book_tags_df: pd.DataFrame,
    # tags_df: pd.DataFrame,
    generalized_tags: pd.DataFrame,
    *,
    min_ratings: int = 5,
    threshold: int = 4,
    negatives_per_pos: int = 4,
    eval_K: int = 10,
    eval_n_neg: int = 99,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Full pipeline following the MFPreprocessor interface:
    1. Uses provided prep (with .fit, .transform, .num_users, .num_items)
    2. Prepares aligned content matrix
    3. Trains NeuMFHybrid
    4. Evaluates on test and train positives
    Returns test/train metrics as dictionary.
    """

    # --- 1. Fit encoders and transform ratings -----------------------------
    prep = MFPreprocessor(
        ratings_df,
        min_ratings=min_ratings,
        threshold=threshold,
        seed=seed,
    )
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

    # Build leave-one-out test positives (assumed encoded)
    num_users = prep.num_users()
    num_items = prep.num_items()

    # --- 2. Build aligned content matrix ----------------------------------
    content_df = build_item_content_matrix(
        generalized_tags=generalized_tags,
        books_df=books_df,
        # book_tags_df=book_tags_df,
        # tags_df=tags_df,
    )
    content_matrix = align_content_matrix(content_df, prep.item_encoder)

    model = train_neumf_hybrid(
        train_df=train_df,
        num_users=num_users,
        num_items=num_items,
        content_matrix=content_matrix,
        epochs=20,
        seed=seed,
    )

    # --- 4. Evaluate on test positives ------------------------------------
    metrics_train = evaluate_topk_hybrid(
        model=model,
        test_df_pos=train_pos,
        train_df_pos=train_pos,
        num_items=num_items,
        content_matrix=content_matrix,
        k=eval_K,
        n_neg=eval_n_neg,
        seed=seed,
    )
    metrics_test = evaluate_topk_hybrid(
        model=model,
        test_df_pos=test_pos,
        train_df_pos=train_pos,
        num_items=num_items,
        content_matrix=content_matrix,
        k=eval_K,
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
