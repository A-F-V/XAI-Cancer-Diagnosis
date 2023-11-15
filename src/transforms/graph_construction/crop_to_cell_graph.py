from src.transforms.graph_construction.node_embedding import generate_node_embeddings
import torch
from src.transforms.image_processing.filters import glcm

# For each crop-graph, compute the following:


def crop_graph_to_cell_graph(crop_graph, resnet_encoder):
    num_nodes = crop_graph.x.shape[0]
    # Generate GLCM for each cell
    cell_graph = crop_graph.clone()
    unflattened_imgs = crop_graph.x.unflatten(1, (3, 64, 64))

    glcm_for_graph = torch.zeros(num_nodes, 50)
    for i, img in enumerate(unflattened_imgs):
        # Unflatten the image
        glcm_for_graph[i] = glcm(img, normalize=True)

    # Pass the cell images through the resnet encoder
    cell_graph.x = generate_node_embeddings(imgs=unflattened_imgs, resnet_encoder=resnet_encoder,
                                            num_neighbours=cell_graph.num_neighbours, cell_types=cell_graph.categories,
                                            glcm=glcm_for_graph.to(device=crop_graph.x.device))
    return cell_graph
