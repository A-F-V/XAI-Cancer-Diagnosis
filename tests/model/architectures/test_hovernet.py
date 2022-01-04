from src.model.architectures.graph_construction.hover_net import HoVerNet, resnet_sizes


def test_hovernet_init():
    for size in resnet_sizes:
        HoVerNet(size)


def test_hovernet_trainable():
    assert True
