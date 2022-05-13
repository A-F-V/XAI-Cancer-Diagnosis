from torch_geometric.data import Data


def show_graph(graph: Data, plot, with_edges=True):
    points = list(map(lambda x: x['centre'], graph))
    x, y = map(list, zip(*points))
    plot.scatter(x, y, c="r")
    if with_edges:
        for node in graph.nodes:
            for neighbour in graph.edges_from[node]:
                p1, p2 = graph.nodes[node]['centre'], graph.nodes[neighbour]['centre']
                plot.plot([p1[0], p2[0]], [p1[1], p2[1]], c="b")
