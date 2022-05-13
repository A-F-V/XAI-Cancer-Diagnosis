from torch.nn.functional import normalize
from tqdm import tqdm
from torch.nn.functional import one_hot
import torch
from sklearn.cluster import MiniBatchKMeans


def nearest_mean(x, means):

    delta = (means - x)**2
    dists = delta.sum(axis=1)
    return dists.argmin()


########################################################################################################################


def graph_to_activations(model, graph):
    # Setup the hook for intercepting activations

    raw_activations = torch.zeros(0, 32)

    def append_raw_activations(self, input, output):
        global raw_activations
        raw_activations = torch.cat((raw_activations, output), dim=0)

    hook = model.gnn.conv[-1].register_forward_hook(append_raw_activations)
    model(graph.x, graph.edge_index, graph.batch)

    hook.remove()
    return raw_activations


def compile_all_activations(model, train_loader):
    activations = []
    for batch in tqdm(train_loader):
        activations.append(graph_to_activations(model, batch))
    return torch.cat(activations, dim=0)


def whiten(obs, mu, sigma):
    return (obs - mu)/sigma


def fetch_activations_and_statistics(model, loader):
    raw_activations = compile_all_activations(model, loader)

    obs = raw_activations.detach().numpy()
    mu, sigma = obs.mean(axis=0), obs.std(axis=0)

    return whiten(obs, mu, sigma), mu, sigma


def generate_concept_means(model, train_loader, k):
    obs_white = fetch_activations_and_statistics(model, train_loader)[0]
    means = MiniBatchKMeans(n_clusters=k, batch_size=10000).fit(obs_white).cluster_centers_
    return means
########################################################################################################################


def single_activation_to_concept(activation, means):
    return one_hot(torch.as_tensor(nearest_mean(activation, means)), len(means))


def activation_to_concepts(activations, means, k):
    output = torch.zeros(0, k)
    for i in range(len(activations)):
        concept = single_activation_to_concept(activations[i].numpy(), means).unsqueeze(0)
        output = torch.cat([output, concept], dim=0)
    return output


def predict_activations_and_concept_from_graph(model, graph, means, k, mu, sigma):
    activations = graph_to_activations(model, graph)
    activations = whiten(activations, mu, sigma)
    return activations, activation_to_concepts(activations, means, k)

########################################################################################################################


def graph_to_activation_concept_graph(model, graph, means, k, mu, sigma):
    output = graph.clone()
    output.activation, output.x = predict_activations_and_concept_from_graph(model, graph, means, k, mu, sigma)
    return output
