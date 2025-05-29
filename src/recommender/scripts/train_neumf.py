import os
from pathlib import Path
import sys
from dotenv import load_dotenv

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from recommender.data.download import DATA_PATH
from recommender.scripts.run_pipeline_neumf import run_pipeline_neumf

load_dotenv()


if __name__ == "__main__":
    ratings_df = pd.read_csv(DATA_PATH.joinpath("ratings.csv"))
    ratings_df_sampled = ratings_df.sample(frac=0.25, random_state=42)

    books_df = pd.read_csv(DATA_PATH.joinpath("books.csv"))

    # Referencing to prj root based on dynamic path since running on aml instance.
    project_root = Path(os.environ["AML_INSTANCE_PATH_PREFIX"])  
    tags_dataset_path = project_root.joinpath(
        'pytorch-MF-NeuMF', 'src', 'recommender', 'datasets', 'generalized_tags.xlsx'
    ).as_posix()

    generalized_tags = pd.read_excel(tags_dataset_path)

    model_neumf, results, _ = run_pipeline_neumf(
        ratings_df=ratings_df_sampled,
        books_df=books_df,
        # book_tags_df=book_tags_df,
        # tags_df=tags_df,
        generalized_tags=generalized_tags,
        min_ratings= 5,
        threshold= 4,
        negatives_per_pos=20,
        eval_K=100,
        eval_n_neg=999,
        seed=42,
    )
