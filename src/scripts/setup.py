from src.scripts.data_scripts.fetch_data import download_undownloaded_dataset
from src.scripts.data_scripts.dataset_manager import data_path_raw_folder, data_path_folder
from src.scripts.data_scripts.extract_data import unzip_dataset
from src.scripts.data_scripts.preprocess_data import move_and_rename, create_semantic_segmentation_mask
import os
from tqdm import tqdm


def setup():
    data_sets = ["MoNuSeg", "BACH"]

    for data_set in data_sets:
        download_undownloaded_dataset(data_set, folder=data_path_raw_folder)
        unzip_dataset(data_set, data_path_raw_folder)

    ###############################################################################
    # Process MoNuSeg                                                             #
    ###############################################################################

    MoNuSeg_unzipped = os.path.join(data_path_raw_folder, "unzipped", "MoNuSeg", "MoNuSeg 2018 Training Data")
    move_and_rename(MoNuSeg_unzipped,
                    {"Annotations": "annotations", "Tissue Images": "images"},
                    os.path.join(data_path_folder, "MoNuSeg"))

    for image_name in tqdm(os.listdir(os.path.join(data_path_folder, "MoNuSeg", "images")), desc="Extracting Annotated Masks"):
        img_path = os.path.join(data_path_folder, "MoNuSeg", "images", image_name)
        anno_path = os.path.join(data_path_folder, "MoNuSeg", "annotations", image_name.split(".")[0] + ".xml")
        dst_folder = os.path.join(data_path_folder, "MoNuSeg", "semantic_masks")
        create_semantic_segmentation_mask(anno_path, img_path, dst_folder)
