import os
from shutil import copyfile
from PIL import Image
import numpy as np
from src.data_processing.semantic_segmentation_mask import SemanticSegmentationMask
from src.data_processing.instance_segmentation_mask import InstanceSegmentationMask


def move_and_rename(src, folders: dict, dst: str):
    """Take files from srcs, rename them to ascending index and place in dst

    Args:
        src (str): Path to folder where files will be move from
        folders (dict): Folder names and the respective new folder name in dst
        dst (str): Path to folder where files will be moved to

    Returns:
        None
    """
    # create index
    a_folder = os.path.join(src, list(folders)[0])
    files = list(map(lambda x: x.split(".")[-2], os.listdir(a_folder)))
    index = {files[i]: i for i in range(len(files))}
    for s_fold, d_fold in folders.items():
        dest_folder_path = os.path.join(dst, d_fold)
        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path,)
        for f in os.listdir(os.path.join(src, s_fold)):
            file = os.path.basename(f)
            file_name = file.split(".")[0]
            extension = file.split(".")[-1]
            old_file_path = os.path.join(src, s_fold, file)
            new_file_path = os.path.join(dest_folder_path, f"{index[file_name]}.{extension}")
            copyfile(old_file_path, new_file_path)


def _semantic_mask(anno_path, size):
    return SemanticSegmentationMask(anno_path).create_mask(filled=True, size=size)


def _instance_mask(anno_path, size):
    return InstanceSegmentationMask(anno_path).create_mask(filled=True, size=size)


def apply_mask_extraction(extraction_function, anno_path, img_path, dst_folder):
    original_image = Image.open(img_path, 'r')
    original_image.load()
    mask = extraction_function(anno_path, size=original_image.size)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    s_img_path = os.path.join(dst_folder, os.path.basename(img_path).split(".")[0]+".npy")
    np.save(s_img_path, mask)


def create_semantic_segmentation_mask(anno_path, img_path, dst_folder):
    """Creates and saves a semantic segmentation mask of img_path from the annotations of anno_path
        The name of the file will be the same as the image_path
    Args:
        anno_path (str): Path to annotation
        img_path (str): Path to image
        dst_folder (str): Path to the folder to store mask.
    """
    apply_mask_extraction(_semantic_mask, anno_path, img_path, dst_folder)


def create_instance_segmentation_mask(anno_path, img_path, dst_folder):
    apply_mask_extraction(_instance_mask, anno_path, img_path, dst_folder)
