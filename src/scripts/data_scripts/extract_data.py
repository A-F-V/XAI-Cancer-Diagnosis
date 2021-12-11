import zipfile
import os
from .dataset_manager import data_path_raw_folder


def unzip_file_to_folder(zip_path, extract_path):
    """Takes a zip file path and extracts the contents to a specified path.

    Args:
        zip_path (str): The location of the zip file
        extract_path (str): The location of the extracted files
    """

    if os.path.exists(extract_path):
        print("Extract path already exists")
        return
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def unzip_dataset(data_set_name, folder):
    """Takes the name of a dataset and extracts the contents

    Args:
        data_set_name (str): The name of the dataset
    """
    print("Extracting dataset: " + data_set_name)
    zip_path = os.path.join(folder, "zipped", data_set_name + ".zip")
    extract_path = os.path.join(folder, "unzipped", data_set_name)
    unzip_file_to_folder(zip_path, extract_path)
    print(f"Extracted {data_set_name}.")
