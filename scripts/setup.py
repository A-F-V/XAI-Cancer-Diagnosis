from data_scripts.fetch_data import download_undownloaded_dataset
from data_scripts.dataset_manager import data_path_folder
from data_scripts.extract_data import unzip_dataset
import os

if not os.path.exists(data_path_folder):
    os.makedirs(data_path_folder)

data_sets = ["MoNuSeg", "BACH"]

for data_set in data_sets:
    download_undownloaded_dataset(data_set)
    unzip_dataset(data_set)
