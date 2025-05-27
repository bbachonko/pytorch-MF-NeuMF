import os
import sys

import pandas as pd

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from recommender.data.download import DATA_PATH
from recommender.scripts.run_pipeline_neumf import run_pipeline_neumf


if __name__ == "__main__":
    ratings_df = pd.read_csv(DATA_PATH.joinpath("ratings.csv"))
    ratings_df_sampled = ratings_df.sample(frac=0.25, random_state=42)

    books_df = pd.read_csv(DATA_PATH.joinpath("books.csv"))
    # generalized_tags = 




model_neumf, results, _ = run_pipeline_neumf(
    ratings_df=ratings_df_sampled,
    books_df=books_df,
    # book_tags_df=book_tags_df,
    # tags_df=tags_df,
    generalized_tags=generalized_tags,
    min_ratings= 5,
    threshold= 4,
    negatives_per_pos= 8,
    eval_K= 10,
    eval_n_neg= 999,
    seed=42,
)