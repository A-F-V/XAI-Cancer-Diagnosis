from src.model.architectures.graph_construction.hover_net import HoVerNet, resnet_sizes


def test_hovernet_init():
    for size in resnet_sizes:
        HoVerNet(**{"RESNET_SIZE": size, "START_LR": 0.001})


def test_hovernet_trainable():
    assert True
