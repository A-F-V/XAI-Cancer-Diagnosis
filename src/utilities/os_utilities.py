import os


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
