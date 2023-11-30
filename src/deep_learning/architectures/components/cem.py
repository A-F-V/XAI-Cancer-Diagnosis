import torch
# https://arxiv.org/pdf/2207.13586.pdf


class ConceptEncoderModule(torch.nn.Module):
    def __init__(self, width, e=1e-10):
        super(ConceptEncoderModule, self).__init__()
        self.width = width
        self.sm = torch.nn.Softmax(dim=1)
        self.epsilon = e

    def forward(self, x):
        (num_nodes, num_features) = x.shape
        assert num_features == self.width

        # Soft max
        q_i_hat = self.sm(x)
        assert q_i_hat.shape == (num_nodes, num_features)

        # Normalize over max of each feature
        node_maxs = torch.max(q_i_hat, dim=1).values
        norm_factor = 1/(node_maxs + self.epsilon)
        norm_factor = norm_factor.reshape(num_nodes, 1)
        assert norm_factor.shape == (num_nodes, 1)
        q_i = norm_factor * q_i_hat
        assert q_i.shape == (num_nodes, num_features)

        return q_i
