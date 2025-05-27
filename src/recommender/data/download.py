from pathlib import Path
import os
import shutil

import kagglehub

from recommender.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_data_from_kagglehub(dataset: str = "zygmunt/goodbooks-10k") -> str:
    src_path = kagglehub.dataset_download(dataset)    
    dst_path = os.path.join(os.getcwd(), "goodbooks-10k")

    logger.info(f"Kaggle Dataset: {dataset} was downloaded properly under path: {dst_path}")

    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    return dst_path


DATA_PATH = Path(load_data_from_kagglehub())