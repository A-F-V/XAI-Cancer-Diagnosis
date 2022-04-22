import torch
from torch.utils.data import Dataset
import os
from src.utilities.os_utilities import create_dir_if_not_exist
from torchvision.utils import save_image
from tqdm import tqdm
from multiprocessing import Pool
from src.algorithm.counting_matrix import CountingMatrix
import pickle
import numpy as np
from torch.nn.functional import one_hot


class BACH_Cells(Dataset):
    def __init__(self, src_folder, img_dim=64, transform=None, ids=None, val=False):
        super(BACH_Cells, self).__init__()
        self.src_folder = src_folder
        self.img_dim = img_dim

        create_dir_if_not_exist(self.cell_dir, False)
        create_dir_if_not_exist(self.cell_val_dir, False)
        create_dir_if_not_exist(self.cell_img_dir, False)

        self.val = val
        self.ids = ids if ids is not None else list(
            range(len([f for f in os.listdir(self.cell_dir if not self.val else self.cell_val_dir) if ".pt" == f[-3:]])))
        self.img_augmentation = transform
        self.hit = 0

    # def compile_cells(self, regenerate=False):
    #    if os.path.exists(self.saved_cm) and not regenerate:
    #        with open(self.saved_cm, 'rb') as f:
    #            self.cm = pickle.load(f)
    #    else:
    #        self.cm = CountingMatrix(len(self.graph_paths))
    #        for i, path in tqdm(enumerate(self.graph_paths)):
    #            data = torch.load(path)
    #            self.cm.add_many(i, data.x.shape[0])
    #        self.cm.cumulate()
    #        pickle.dump(self.cm, open(self.saved_cm, 'wb'))

    def compile_cells(self, recompute=False, train_test_split=1.0):
        n = 0
        #print("Finding Cells")
        num_cells = len([os.path.join(self.cell_dir, f) for f in os.listdir(self.cell_dir) if ".pt" == f[-3:]])
        #print("Found {} cells".format(num_cells))
        if not recompute and num_cells != 0:
            self.num_cells = num_cells
        else:
            train_ind, val_ind = [], []
            for clss in range(4):
                random_ids = np.arange(clss*100, (clss+1)*100)
                np.random.shuffle(random_ids)
                train_ind += list(random_ids[:int(100*train_test_split)])
                val_ind += list(random_ids[int(100*train_test_split):])
            nt, nv = 0, 0
            for i, graph_path in tqdm(enumerate(self.graph_paths), total=len(self.graph_paths)):
                graph = torch.load(graph_path)
                x = graph.x
                y = graph.y
                categories = graph.categories
                start_id = nt if i in train_ind else nv
                for cell_id in range(start_id, start_id+x.shape[0]):
                    relative_id = cell_id - start_id
                    cell = x[relative_id].unflatten(0, (3, self.img_dim, self.img_dim)).clone()
                    cell_type = categories[relative_id].int().item()  # background = 0
                    cell_type_one_hot = torch.zeros(5)
                    cell_type_one_hot[cell_type] = 1
                    assert cell.shape == (3, 64, 64)
                    assert y.shape == (4,)
                    assert cell_type_one_hot.shape == (5,)
                    torch.save({'img': cell, 'diagnosis': y, 'cell_type': cell_type_one_hot}, (os.path.join(
                        self.cell_dir if i in train_ind else self.cell_val_dir, f'{cell_id}.pt')))

                    save_image(cell, os.path.join(self.cell_img_dir, f'{cell_id}.png'))
                nt, nv = (nt+x.shape[0]) if i in train_ind else nt, (nv+x.shape[0]) if i in val_ind else nv
                n += x.shape[0]
            f = open(os.path.join(self.src_folder, "graph_ind.txt"), "w")
            f.write(str(train_ind)+"\n"+str(val_ind))
            f.close()
            self.num_cells = n

    @property
    def cell_img_dir(self):
        return os.path.join(self.src_folder, "CELL_CROPS")

    @property
    def cell_val_dir(self):
        return os.path.join(self.src_folder, "CELLS_VAL")

    @property
    def cell_dir(self):
        return os.path.join(self.src_folder, "CELLS")

    @property
    def saved_cm(self):
        return os.path.join(self.graph_dir, "CellCounts.pkl")

    @property
    def graph_dir(self):
        return os.path.join(self.src_folder, "GRAPH")

    @property
    def graph_file_names(self):
        return [f for f in os.listdir(self.graph_dir) if ".pt" in f]

    @property
    def graph_paths(self):
        return sorted([os.path.join(self.graph_dir, f) for f in self.graph_file_names])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        # (graph_idx, cell_idx) = self.cm[self.ids[ind]]
        # graph = torch.load(self.graph_paths[graph_idx])
        # cell = graph.x[cell_idx].unflatten(0, (3, self.img_dim, self.img_dim))
        # y = graph.y
        # if self.img_augmentation is not None:
        #    cell = self.img_augmentation(cell)
        # return {'img': cell, "diagnosis": y}
        path = os.path.join(self.cell_dir if not self.val else self.cell_val_dir, str(self.ids[ind])+".pt")
        data = torch.load(path)
        if self.img_augmentation is not None:
            data = self.img_augmentation(data)  # BECAUSE TRANSFORMS ACT ON ENTIRE PAYLOAD
        cell, y, cell_type = data['img'], data['diagnosis'], data['cell_type']
        return {'img': cell, "diagnosis": y, "cell_type": cell_type}
