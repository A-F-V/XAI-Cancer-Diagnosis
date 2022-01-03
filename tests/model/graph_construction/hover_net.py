from src.model.graph_construction.hover_net import HoVerNet


def test_hover_net_compiles():
    model = HoVerNet()
    assert model is not None
