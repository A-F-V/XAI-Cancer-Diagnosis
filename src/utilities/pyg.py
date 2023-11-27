from torch_geometric.data import Batch, Data


def loader_from_one_graph(graph: Data):
    return DataLoader([graph], batch_size=1, shuffle=False)


def extract_graph(batch: Batch, graph_idx: int):
    # Find node indices for the graph
    node_mask = batch.batch == graph_idx

    # Extract the node features for the graph
    x = batch.x[node_mask]

    # Find edge indices for the graph
    edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
    edge_index = batch.edge_index[:, edge_mask]

    # Re-map edge indices to the new node index space
    edge_index = edge_index - node_mask.nonzero(as_tuple=False).min()

    # If the batch contains other attributes, extract them similarly
    # ...
    y = batch.y[graph_idx]
    pos = batch.pos[node_mask]

    # Create a new Data object for the single graph
    single_graph = Data(x=x, edge_index=edge_index, y=y, pos=pos)
    single_graph.original_image_path = batch.original_image_path[graph_idx]

    return single_graph


def batch_to_graphs(batch: Batch):
    num_graphs = batch.batch.max().item()
    for ind in range(num_graphs):
        yield extract_graph(batch, ind)
