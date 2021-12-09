from data_scripts.fetch_data import download_undownloaded_dataset, data_path_folder
import os

# Get the current working directory
print(data_path_folder)


if not os.path.exists(data_path_folder):
    os.makedirs(data_path_folder)

download_undownloaded_dataset("MoNuSeg")
download_undownloaded_dataset("BACH")
