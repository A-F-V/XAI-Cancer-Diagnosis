class Graph:
    def __init__(self, directed=False):
        self.nodes = {}
        self.edges_from = {}
        self.edges_to = {}
        self.edge_val = {}
        self.directed = directed
        self._node_count = 0
        self._edge_count = 0

    def add_node(self, node):
        node_id = len(self.nodes)
        self.nodes[node_id] = node
        if node_id not in self.edges_from:
            self.edges_from[node_id] = set()
        if node_id not in self.edges_to:
            self.edges_to[node_id] = set()
        self._node_count += 1

    def add_edge(self, fr_node, to_node, edge):
        self.edges_from[fr_node].add(to_node)
        self.edges_to[to_node].add(fr_node)
        self.edge_val[(fr_node, to_node)] = edge
        self._edge_count
        if not self.directed:
            self.edges_from[to_node].add(fr_node)
            self.edges_to[fr_node].add(to_node)
            self.edge_val[(to_node, fr_node)] = edge

    def __len__(self):
        return self._node_count, self._edge_count
