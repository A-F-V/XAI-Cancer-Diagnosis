import torch


def hard_agreement(x, edge_index):
    dec = x.argmax(dim=1)
    num_nodes = dec.shape[0]
    agree, nei = torch.zeros(num_nodes), torch.zeros(num_nodes)
    for ni, nj in edge_index:
        nei[ni] += 1
        agree[ni] += 1 if dec[ni] == dec[nj] else 0
    return (agree/nei).mean()
