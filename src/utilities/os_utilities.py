import os
import shutil


def create_dir_if_not_exist(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def copy_dir(src, dst):  # copies and replaces
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_file(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    create_dir_if_not_exist(dst)
    shutil.copy(src, dst)
