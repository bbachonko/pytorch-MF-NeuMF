import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from recommender.data.download import DATA_PATH
from recommender.scripts.run_pipeline_mf import run_pipeline


if __name__ == "__main__":
    ratings_df = pd.read_csv(DATA_PATH.joinpath("ratings.csv"))
    ratings_df_sampled = ratings_df.sample(frac=0.25, random_state=42)

    model, results, _ = run_pipeline(
        ratings=ratings_df_sampled,
        min_ratings= 5,
        threshold= 4,
        negatives_per_pos= 8,
        eval_K= 10,
        eval_n_neg= 999,
        seed=42,
    )
