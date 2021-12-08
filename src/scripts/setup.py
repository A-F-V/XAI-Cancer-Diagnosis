

from data_scripts.fetch_data import download_undownloaded_dataset
import os

# Get the current working directory
cwd = os.getcwd()

print(cwd)
download_undownloaded_dataset("MoNuSeg.zip", "MoNuSeg.zip")
