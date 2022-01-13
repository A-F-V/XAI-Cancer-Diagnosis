import os
import shutil


def create_dir_if_not_exist(path, file_path=True):
    """Creates a directory if does not exist.

    Args:
        path (str): The path
        file_path (bool, optional): If True, get the directory of the file provided. Defaults to True.
    """
    if file_path:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


def copy_dir(src, dst):  # copies and replaces
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_file(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    create_dir_if_not_exist(dst)
    shutil.copy(src, dst)
