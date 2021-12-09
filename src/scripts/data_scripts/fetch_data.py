import os
from google.cloud import storage
from tqdm import tqdm

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secrets/google_credentials.json"
storage_client = storage.Client()
my_bucket = storage_client.get_bucket("medical-dataset-xai-cancer-diagnosis")

data_path_folder = os.path.join(os.getcwd(), "data", "datasets", "raw")


def download_dataset(blob, filepath):
    """Downloads a dataset from a given blob to a certain path

    Args:
        blob
        filepath
    """
    with open(filepath, "wb") as f:
        with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
            storage_client.download_blob_to_file(blob, file_obj)


def download_undownloaded_dataset(data_set):
    """Downloads a dataset from the google bucket if it is not already downloaded
    Args:
        data_set (str): Name of the data_set
    """
    print("Downloading dataset: " + data_set)
    blob = my_bucket.get_blob(data_set+".zip")
    filepath = os.path.join("data", "datasets", "raw", data_set+".zip")
    if os.path.exists(filepath) and os.path.getsize(filepath) == blob.size:
        print(data_set + " already donwloaded")
    else:
        download_dataset(blob, filepath)
        print(data_set + " downloaded")
