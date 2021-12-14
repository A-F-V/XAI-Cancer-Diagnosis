from src.model.graph_construction.graph import Graph


def show_graph(graph: Graph, plot):
    """
    Shows the graph using graphviz.
    """
    points = list(map(lambda x: x['centre'], graph.nodes.values()))
    x, y = map(list, zip(*points))
    plot.scatter(x, y, c="r")
    for node in graph.nodes:
        for neighbour in graph.edges_from[node]:
            p1, p2 = graph.nodes[node]['centre'], graph.nodes[neighbour]['centre']
            plot.plot([p1[0], p2[0]], [p1[1], p2[1]], c="b")
