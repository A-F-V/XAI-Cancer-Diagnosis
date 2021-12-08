import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secrets/google_credentials.json"
storage_client = storage.Client()
my_bucket = storage_client.get_bucket("medical-dataset-xai-cancer-diagnosis")


def download_dataset(blob_name, file_name):
    blob = my_bucket.blob(blob_name)
    blob.download_to_filename(os.path.join("data", "datasets", "raw", file_name))


def download_undownloaded_dataset(blob_name, file_name):
    print("Downloading dataset: " + blob_name)
    if os.path.exists(os.path.join("data", "datasets", "raw", file_name)):
        print(blob_name + " already donwloaded")
    else:
        download_dataset(blob_name, file_name)
