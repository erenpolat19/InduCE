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
import torch

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
    start = batch.ptr
