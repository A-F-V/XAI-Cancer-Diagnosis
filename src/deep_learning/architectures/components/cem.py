import torch
# https://arxiv.org/pdf/2207.13586.pdf


class ConceptEncoderModule(torch.nn.Module):
    def __init__(self, width, e=0.00001):
        super(ConceptEncoderModule, self).__init__()
        self.width = width
        self.sm = torch.nn.Softmax(dim=1)
        self.epsilon = e

    def forward(self, x, edge_index, batch):
        (num_nodes, num_features) = x.shape
        assert num_features == self.width

        # Soft max
        q_i_hat = self.sm(x)
        assert q_i_hat.shape == (num_nodes, num_features)

        # Normalize over max of each feature
        q_i = q_i_hat / (torch.max(q_i_hat, dim=1).values + self.epsilon)
        assert q_i.shape == (num_nodes, num_features)

        return q_i
