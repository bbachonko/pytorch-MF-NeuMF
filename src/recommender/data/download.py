import kagglehub
import os
import shutil


def load_data_from_kagglehub(dataset: str = "zygmunt/goodbooks-10k") -> str:
    src_path = kagglehub.dataset_download(dataset)
    dst_path = os.path.join(os.getcwd(), "goodbooks-10k")
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    return dst_path