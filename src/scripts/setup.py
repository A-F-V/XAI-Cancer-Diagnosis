
from src.scripts.data_scripts.fetch_data import download_undownloaded_dataset
from src.scripts.data_scripts.dataset_manager import data_path_raw_folder, data_path_folder
from src.scripts.data_scripts.extract_data import unzip_dataset
from src.scripts.data_scripts.preprocess_data import move_and_rename, create_semantic_segmentation_mask
from src.transforms.he_normalize import normalize_he_image
from src.utilities.img_utilities import *
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os
from tqdm import tqdm
import numpy as np

from src.utilities.os_utilities import copy_dir


def setup():
    data_sets = ["MoNuSeg_TRAIN", "BACH_TRAIN", "BACH_TEST", "MoNuSeg_TEST", "PanNuke"]

    unzipped_folder = os.path.join(data_path_raw_folder, "unzipped")

    for data_set in data_sets:
        try:
            download_undownloaded_dataset(data_set, folder=data_path_raw_folder)
        except:
            print("Failed to download: " + data_set)
        try:
            unzip_dataset(data_set, data_path_raw_folder)
        except:
            print("Failed to unzip: " + data_set)
    ###############################################################################
    # Process MoNuSeg                                                             #
    ###############################################################################

# todo incorporate test as well. Have both in same folder (N)

    MoNuSeg_unzipped = os.path.join(data_path_raw_folder, "unzipped", "MoNuSeg_TRAIN", "MoNuSeg 2018 Training Data")
    move_and_rename(MoNuSeg_unzipped,
                    {"Annotations": "annotations", "Tissue Images": "images"},
                    os.path.join(data_path_folder, "MoNuSeg_TRAIN"))

    for image_name in tqdm(os.listdir(os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images")), desc="Extracting Annotated Masks - MoNuSeg"):
        img_path = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images", image_name)
        anno_path = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "annotations", image_name.split(".")[0] + ".xml")
        dst_folder = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "semantic_masks")
        create_semantic_segmentation_mask(anno_path, img_path, dst_folder)

    for image_name in tqdm(os.listdir(os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images")), desc="Normalizing Images - MoNuSeg"):
        img_path = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images", image_name)
        img = Image.open(img_path)
        img = normalize_he_image(ToTensor()(img))
        img = ToPILImage()(img)
        img.save(img_path)

        ###############################################################################
    # Process PanNuke                                                             #
    ###############################################################################

    copy_dir(os.path.join(unzipped_folder, "PanNuke"), os.path.join(data_path_folder, "PanNuke"))

    # NORM IMAGES - NO LONGER AS TOO SMALL TO DO WELL.

    #images = np.load(os.path.join(data_path_folder, "PanNuke", "images.npy"))
    # def safe_norm(img):
    #    try:
    #        return tensor_to_numpy(normalize_he_image(numpy_to_tensor(img)))
    #    except:
    #        return img
    # norm_images = [safe_norm(img)
    #               for img in tqdm(images, desc="Normalizing Images - PanNuke")]
    #norm_images = np.stack(norm_images, axis=0)
    # if norm_images.max() <= 1:
    #    norm_images *= 255
    #norm_images = norm_images.astype(np.uint8)
    #np.save(os.path.join(data_path_folder, "PanNuke", "images.npy"), norm_images)
