from src.algorithm.counting_matrix import CountingMatrix


def test_counting_matrix():
    cm = CountingMatrix(4)
    cm.add(0)
    cm.add_many(1, 3)
    cm.add_many(2, 4)
    cm.add_many(3, 2)

    assert cm[0] == (0, 0)
    assert cm[3] == (1, 2)
    assert cm[8] == (3, 0)
    assert len(cm) == 10

    try:
        cm[-1]
        assert False
    except IndexError:
        assert True

    try:
        cm[10]
        assert False
    except IndexError:
        assert True
