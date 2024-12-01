
from data.data import * 
from math import floor
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch
import random
from torch_geometric.utils import dense_to_sparse
from neurosed import models
'''
Load neurosed model to use it for ged calculation in the reward

Take two instances from AIDS dataset, G_input and G_target

Put them in those Graph objects in the Induce code

Numnodes x numnodes -> probability    (u1, u2)   

Potentially we extend induce (idk if it already has it) to be adding nodes of different kinds? 

'''

def split_data(dataset, valid_ratio=0.1, test_ratio=0.1):
    seed = 1
    generator = torch.Generator().manual_seed(seed)
    valid_size = floor(len(dataset) * valid_ratio)
    test_size = floor(len(dataset) * test_ratio)
    train_size = len(dataset) - valid_size - test_size
    splits = torch.utils.data.random_split(dataset, lengths=[train_size, valid_size, test_size], generator=generator)

    return splits

def extract_graph_from_batch(batch, graph_idx):
    start = batch.ptr[graph_idx].item()
    end = batch.ptr[graph_idx + 1].item()

    x = batch.x[start:end]  
    edge_index = batch.edge_index[:, batch.edge_index[0] >= start]
    edge_index = edge_index[:, edge_index[1] < end]  

    if batch.edge_attr is not None:
        edge_attr = batch.edge_attr[batch.edge_index[0] >= start]
        edge_attr = edge_attr[edge_index[1] < end]  
    else:
        edge_attr = None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=batch.y[graph_idx:graph_idx+1])

    return data

def add_edges(batch, graph_idx, edges_to_add, default_edge_attr=None):
    start = batch.ptr[graph_idx].item()
    end = batch.ptr[graph_idx + 1].item()

    adjusted_edges = [(start + u, start + v) for u, v in edges_to_add]
    new_edges = torch.tensor(adjusted_edges, dtype=torch.long).t()

    updated_edge_index = torch.cat([batch.edge_index, new_edges], dim=1)

    if batch.edge_attr is not None:
        num_new_edges = new_edges.size(1)

        if default_edge_attr is not None:
            new_edge_attrs = torch.tensor([default_edge_attr] * num_new_edges, dtype=batch.edge_attr.dtype)
        else:
            new_edge_attrs = torch.zeros(num_new_edges, dtype=batch.edge_attr.dtype)  
        
        updated_edge_attr = torch.cat([batch.edge_attr, new_edge_attrs], dim=0)
    else:
        updated_edge_attr = None

    updated_batch = Batch(
        x=batch.x,
        edge_index=updated_edge_index,
        edge_attr=updated_edge_attr,
        y=batch.y,
        num_nodes=batch.num_nodes,
        batch=batch.batch,
        ptr=batch.ptr,
    )

    return updated_batch

def visualize_graph_from_batch(batch, graph_idx=0, title="Graph"):
    start = batch.ptr[graph_idx].item()
    end = batch.ptr[graph_idx + 1].item()

    data = Data(
        x=batch.x[start:end],
        edge_index=batch.edge_index[:, (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)] - start,
        edge_attr=batch.edge_attr[(batch.edge_index[0] >= start) & (batch.edge_index[0] < end)],
        y=batch.y[graph_idx],
        num_nodes=end - start,
    )
    G = to_networkx(data, edge_attrs=['edge_attr'], node_attrs=['x'])

    node_colors = "skyblue" 

    if 'edge_attr' in data:
        edge_labels = {(u, v): f"{d['edge_attr']:.2f}" for u, v, d in G.edges(data=True)}
    else:
        edge_labels = None

    pos = nx.spring_layout(G) 

    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
        cmap=plt.cm.Blues,
        node_size=100,
        font_size=10,
    )

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.savefig(f"{title}.png")

# def create_graph_from_data(data):
#     x = data.x
#     adj = torch.zeros((data.num_nodes, data.num_nodes))
#     edge_index = data.edge_index
#     adj[edge_index[0], edge_index[1]] = 1  
    
#     edge_index, _ = dense_to_sparse(adj)

#     row_sum = adj.sum(dim=1)
#     norm_adj = adj / row_sum.unsqueeze(-1)
    
#     return Graph(x, adj, data.y, None, None, edge_index, norm_adj)

# class Graph:
#     def __init__(self, x, adj, labels, idx_train, idx_test, edge_index, norm_adj):
#         self.x = x
#         self.adj = adj         
#         self.labels = labels  
#         self.idx_train = idx_train 
#         self.idx_test = idx_test   
#         self.edge_index = edge_index  
#         self.norm_adj = norm_adj 

def graph_neurosed_distance_all(graphs, model: models.NormGEDModel):
    batch_size = len(graphs) * len(graphs)
    while True:
        try:
            res = model.predict_outer(graphs, graphs, batch_size=batch_size)
            break
        except RuntimeError as re:
            batch_size = batch_size // 2
    return res


def load_neurosed(original_graphs, neurosed_model_path, device):
    """
    Returns model and embed original graphs.

    :param original_graphs: PyG datasets
    :param neurosed_model_path: loading path of model
    :param device: cuda device to load, or 'cpu'
    :return:
    """
    if not os.path.exists(neurosed_model_path):
        raise FileNotFoundError(f'The neurosed model: {neurosed_model_path} is not found!')

    model = models.NormGEDModel(8, original_graphs[0].x.shape[1], 64, 64, device=device)
    model.load_state_dict(torch.load(neurosed_model_path, map_location=device))
    model.eval()
    model.embed_targets(original_graphs)

    return model

if __name__ == "__main__":
    dataset = load_dataset('aids')

    train_set, valid_set, test_set = split_data(dataset)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=1)
    test_loader = DataLoader(test_set, batch_size=1)


    data_list = []
    for data in train_loader:
        data_list.append(data)

    print(data_list[0])
    batch = data_list[0]

    graph_idx = 0  
    visualize_graph_from_batch(batch, title="Original Graph")
    g_input = extract_graph_from_batch(batch, graph_idx)
    print(g_input)

    new_edges = [(1, 8), (3, 6), (2, 7)]
    new_batch = add_edges(batch, 0, new_edges)
    visualize_graph_from_batch(new_batch, title='Edited Graph')
    g_target = extract_graph_from_batch(new_batch, graph_idx)
    print(g_target)

    # g_input = create_graph_from_data(g_input)
    # g_target = create_graph_from_data(g_target)

    neurosed_model = load_neurosed([g_input], neurosed_model_path=f'pretrained_neurosed_models/aids/best_model.pt', device='cpu')
    S = neurosed_model.predict_outer_with_queries([g_target], batch_size=1).cpu()
    print("S: ", S)

'random pairs??? visualize? the nodes should be same '


'manually check if node features change anything, does neurosed consider atom types as a part of the cost (distance)'


