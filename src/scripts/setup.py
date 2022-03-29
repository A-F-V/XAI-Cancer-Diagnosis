
import torch
from src.scripts.data_scripts.fetch_data import download_undownloaded_dataset
from src.scripts.data_scripts.dataset_manager import data_path_raw_folder, data_path_folder
from src.scripts.data_scripts.extract_data import unzip_dataset
from src.scripts.data_scripts.preprocess_data import create_instance_segmentation_mask, move_and_rename, create_semantic_segmentation_mask
from src.transforms.graph_construction.hover_maps import hover_map
from src.transforms.image_processing.he_normalize import normalize_he_image
from src.utilities.img_utilities import *
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os
from tqdm import tqdm
import numpy as np
from src.datasets.PanNuke import PanNuke

from src.utilities.os_utilities import copy_dir, copy_file


def setup():
    data_sets = ["MoNuSeg_TRAIN", "BACH_TRAIN", "BACH_TEST", "MoNuSeg_TEST", "PanNuke", "PanNuke_orig"]

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
    # Process BACH_TRAIN                                                            #
    ###############################################################################

    BACH_Train_folder_final = os.path.join(data_path_folder, "BACH_TRAIN")

    copy_dir(os.path.join(unzipped_folder, "BACH_TRAIN", "ICIAR2018_BACH_Challenge",
                          "Photos"), BACH_Train_folder_final)

    # STAIN NORMALIZE
    for folder in ["Benign", "InSitu", "Invasive", "Normal"]:
        folder_path = os.path.join(BACH_Train_folder_final, folder)
        for image_name in tqdm(os.listdir(folder_path), desc=f"Normalizing {folder} Images - BACH_TRAIN"):
            img_path = os.path.join(folder_path, image_name)
            if ".tif" not in img_path:
                os.remove(img_path)
            else:
                continue  # NO STAIN NORMING
                img = Image.open(img_path)
                img = normalize_he_image(ToTensor()(img), alpha=1, beta=0.15)
                img = ToPILImage()(img)
                img.save(img_path)

    ###############################################################################
    # Process MoNuSeg_TRAIN                                                            #
    ###############################################################################

# todo incorporate test as well. Have both in same folder (N)

    MoNuSeg_train_unzipped = os.path.join(data_path_raw_folder, "unzipped",
                                          "MoNuSeg_TRAIN", "MoNuSeg 2018 Training Data")
    move_and_rename(MoNuSeg_train_unzipped,
                    {"Annotations": "annotations", "Tissue Images": "images"},
                    os.path.join(data_path_folder, "MoNuSeg_TRAIN"))

    for image_name in tqdm(os.listdir(os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images")), desc="Extracting Annotated Masks - MoNuSeg_TRAIN"):
        img_path = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images", image_name)
        anno_path = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "annotations", image_name.split(".")[0] + ".xml")
        dst_folder_sm = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "semantic_masks")
        dst_folder_im = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "instance_masks")
        create_semantic_segmentation_mask(anno_path, img_path, dst_folder_sm)
        create_instance_segmentation_mask(anno_path, img_path, dst_folder_im)

    # for image_name in tqdm(os.listdir(os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images")), desc="Normalizing Images - MoNuSeg_TRAIN"):
    #    img_path = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images", image_name)
    #    img = Image.open(img_path)
    #    img = normalize_he_image(ToTensor()(img), alpha=1, beta=0.15)
    #    img = ToPILImage()(img)
    #    img.save(img_path)

        ###############################################################################
        # Process MoNuSeg_TEST                                                            #
        ###############################################################################
# todo incorporate test as well. Have both in same folder (N)

    MoNuSeg_test_unzipped = os.path.join(data_path_raw_folder, "unzipped",
                                         "MoNuSeg_TEST", "MoNuSegTestData")

    MoNuSeg_test_final = os.path.join(data_path_folder, "MoNuSeg_TEST")
    for i, name in enumerate(set(map(lambda x: x.split(".")[0], os.listdir(MoNuSeg_test_unzipped)))):
        copy_file(os.path.join(MoNuSeg_test_unzipped, name + ".tif"),
                  os.path.join(MoNuSeg_test_final, "images", str(i) + ".tif"))
        copy_file(os.path.join(MoNuSeg_test_unzipped, name + ".xml"),
                  os.path.join(MoNuSeg_test_final, "annotations", str(i) + ".xml"))

    for image_name in tqdm(os.listdir(os.path.join(MoNuSeg_test_final, "images")), desc="Extracting Annotated Masks - MoNuSeg_TEST"):
        img_path = os.path.join(MoNuSeg_test_final, "images", image_name)
        anno_path = os.path.join(MoNuSeg_test_final, "annotations", image_name.split(".")[0] + ".xml")
        dst_folder_sm = os.path.join(MoNuSeg_test_final, "semantic_masks")
        dst_folder_im = os.path.join(MoNuSeg_test_final, "instance_masks")
        create_semantic_segmentation_mask(anno_path, img_path, dst_folder_sm)
        create_instance_segmentation_mask(anno_path, img_path, dst_folder_im)

    # for image_name in tqdm(os.listdir(os.path.join(MoNuSeg_test_final, "images")), desc="Normalizing Images - MoNuSeg_TEST"):
    #    img_path = os.path.join(MoNuSeg_test_final, "images", image_name)
    #    img = Image.open(img_path)
    #    img = normalize_he_image(ToTensor()(img), alpha=1, beta=0.15)
    #    img = ToPILImage()(img)
    #    img.save(img_path)

        ###############################################################################
    # Process PanNuke                                                             #
    ###############################################################################

    PanNuke.prepare(os.path.join(unzipped_folder, 'PanNuke_orig'), os.path.join(data_path_folder, "PanNuke"))

    # NORM IMAGES - PanNuke

    images = np.load(os.path.join(data_path_folder, "PanNuke", "images.npy"))

    def safe_norm(img):
        try:
            return tensor_to_numpy(normalize_he_image(numpy_to_tensor(img), alpha=1, beta=0.15))
        except:
            return img

    # norm_images = [safe_norm(img)
    #               for img in tqdm(images, desc="Normalizing Images - PanNuke")]
    #norm_images = np.stack(norm_images, axis=0)
    # if norm_images.max() <= 1:
    #    norm_images *= 255
    #norm_images = norm_images.astype(np.uint8)
    np.save(os.path.join(data_path_folder, "PanNuke", "images.npy"), images)  # norm_images)

    # GENERATE HOVER MAPS
    masks = np.load(os.path.join(data_path_folder, "PanNuke", "masks.npy"))
    # last dim is mask channels, last channel is instance mask
    hv_maps = [hover_map(mask[:, :, -1].astype("int16"))
               for mask in tqdm(masks, desc="Generating HoVer Maps - PanNuke")]
    hv_maps = torch.stack(hv_maps).numpy()
    np.save(os.path.join(data_path_folder, "PanNuke", "hover_maps.npy"), hv_maps)
