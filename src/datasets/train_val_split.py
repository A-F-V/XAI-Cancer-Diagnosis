import numpy as np


def train_val_split(dataset, src_folder, ratio, tr_trans, val_trans, class_sizes=None):
    size = len(dataset(src_folder, None, None))

    random_ids = np.arange(size)
    np.random.shuffle(random_ids)
    train_ids, val_ids = list(random_ids[:int(size * ratio)]), list(random_ids[int(size * ratio):])

    train_set = dataset(src_folder=src_folder, transform=tr_trans, ids=train_ids)
    val_set = dataset(src_folder=src_folder, transform=val_trans, ids=val_ids)
    return train_set, val_set
