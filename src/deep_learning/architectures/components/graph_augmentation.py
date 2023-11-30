import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj


class NodePerturbation:
    def __init__(self, perturb_rate: float, factor: float):
        '''
        Args:
            perturb_rate: Probability of perturbing a node feature
            factor: Standard deviation factor of the noise
        '''
        self.perturb_rate = perturb_rate
        self.factor = factor

    def __call__(self, data: Data):
        num_nodes, num_features = data.x.shape
        std_devs = data.x.std(dim=0)
        mask = torch.rand(num_nodes, num_features) < self.perturb_rate
        noise = torch.randn(num_nodes, num_features) * self.factor*std_devs
        data.x = data.x + mask.type_as(data.x) * noise
        return data


class EdgePerturbation:
    def __init__(self, drop_rate: float, add_rate: float):
        self.drop_rate = drop_rate
        self.add_rate = add_rate

    def __call__(self, data: Data):
        # Dropping edges
        edge_index, _ = dropout_adj(data.edge_index, p=self.drop_rate)
        data.edge_index = edge_index

        # Adding edges
        num_nodes = data.x.size(0)
        num_add = int(data.edge_index.size(1) * self.add_rate)
        new_edges = torch.randint(0, num_nodes, (2, num_add), dtype=torch.long)
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

        return data
