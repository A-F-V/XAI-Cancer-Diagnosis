from src.scripts.data_scripts.fetch_data import download_undownloaded_dataset
from src.scripts.data_scripts.dataset_manager import data_path_raw_folder, data_path_folder
from src.scripts.data_scripts.extract_data import unzip_dataset
from src.scripts.data_scripts.preprocess_data import move_and_rename, create_semantic_segmentation_mask
from src.transforms.he_normalize import normalize_he_image
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os
from tqdm import tqdm


def setup():
    data_sets = ["MoNuSeg_TRAIN", "BACH_TRAIN", "BACH_TEST", "MoNuSeg_TEST"]

    for data_set in data_sets:
        download_undownloaded_dataset(data_set, folder=data_path_raw_folder)
        unzip_dataset(data_set, data_path_raw_folder)

    ###############################################################################
    # Process MoNuSeg                                                             #
    ###############################################################################

# todo incorporate test as well. Have both in same folder (N)

    MoNuSeg_unzipped = os.path.join(data_path_raw_folder, "unzipped", "MoNuSeg_TRAIN", "MoNuSeg 2018 Training Data")
    move_and_rename(MoNuSeg_unzipped,
                    {"Annotations": "annotations", "Tissue Images": "images"},
                    os.path.join(data_path_folder, "MoNuSeg_TRAIN"))

    for image_name in tqdm(os.listdir(os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images")), desc="Extracting Annotated Masks"):
        img_path = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images", image_name)
        anno_path = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "annotations", image_name.split(".")[0] + ".xml")
        dst_folder = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "semantic_masks")
        create_semantic_segmentation_mask(anno_path, img_path, dst_folder)

    for image_name in tqdm(os.listdir(os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images")), desc="Normalizing Images"):
        img_path = os.path.join(data_path_folder, "MoNuSeg_TRAIN", "images", image_name)
        img = Image.open(img_path)
        img = normalize_he_image(ToTensor()(img))
        img = ToPILImage()(img)
        img.save(img_path)
