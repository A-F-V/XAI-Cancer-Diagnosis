import os
import shutil


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_dir(src, dst):  # copies and replaces
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
