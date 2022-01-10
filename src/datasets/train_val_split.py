import numpy as np


def train_val_split(dataset, src_folder, ratio, tr_trans, val_trans):
    size = len(dataset(src_folder, None, None))

    random_ids = np.random.shuffle(range(size))
    train_ids, val_ids = random_ids[:int(size * ratio)], random_ids[int(size * ratio):]

    train_set = dataset(src_folder=src_folder, transform=tr_trans, ids=train_ids)
    val_set = dataset(src_folder=src_folder, transform=val_trans, ids=val_ids)
    return train_set, val_set
