'''
Loader for dataset of this thesis
'''
import os
import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph
from graphgym.config import cfg
from graphgym.register import register_loader
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm


def load_generic_dataset(format, name, dataset_dir):
    all_files = [x for x in sorted(os.listdir(dataset_dir)) if x.endswith('.txt')]
    assert len(all_files) == 10
    assert all(x.endswith('.txt') for x in all_files)

    edge_index_lst = list()
    all_files = sorted(all_files)
    g_all = []
    nodes_path = os.path.join(dataset_dir.replace('1.format', 'nodes_set'), 'nodes.csv')
    nodes_set = pd.read_csv(nodes_path, names=['node'])
    nodes_list_raw = nodes_set['node'].tolist()
    num_nodes = len(nodes_list_raw)
    node_indices = np.sort(np.unique(nodes_list_raw))
    for graph_file in tqdm(all_files):
        graph_file_edge = os.path.join(dataset_dir, graph_file)
        src, dst = list(), list()
        nodes = list()
        with open(graph_file_edge, 'r') as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue
                line = line.strip('\n')
                v1, v2 = line.split(' ')
                src.append(int(v1))
                dst.append(int(v2))

        edge_index = np.stack((src, dst))
        edge_index_lst.append(edge_index)

        # encode edges indices
        enc = OrdinalEncoder(categories=[node_indices, node_indices])
        edge_index = enc.fit_transform(edge_index_lst[-1].transpose()).transpose()
        edge_index = edge_index.astype(int)
        # Add all edges in opposite direction as well.
        edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
        print(edge_index[:, 0], edge_index[:, len(dst)])
        edge_index = torch.Tensor(edge_index).long()

        assert cfg.dataset.AS_node_feature in ['one', 'one_hot_id',
                                               'one_hot_degree_global',
                                               'one_hot_degree_local']

        if cfg.dataset.AS_node_feature == 'one':
            node_feature = torch.ones(num_nodes, len(cfg.transaction.feature_node_int_num))
        elif cfg.dataset.AS_node_feature == 'one_hot_id':
            # One hot encoding the node ID.
            node_feature = torch.Tensor(np.eye(num_nodes))
        elif cfg.dataset.AS_node_feature == 'one_hot_degree_global':
            # undirected graph, use only out degree.
            _, node_degree = torch.unique(edge_index[0], sorted=True,
                                          return_counts=True)
            node_feature = np.zeros((num_nodes, node_degree.max() + 1))
            node_feature[np.arange(num_nodes), node_degree] = 1
            # 1 ~ 63748 degrees, but only 710 possible levels, exclude all zero
            # columns.
            non_zero_cols = (node_feature.sum(axis=0) > 0)
            node_feature = node_feature[:, non_zero_cols]
            node_feature = torch.Tensor(node_feature)
        else:
            raise NotImplementedError

        edge_feature = torch.ones(edge_index.size(1), 1)

        g_all.append(Graph(
            node_feature=node_feature,
            edge_index=edge_index,
            edge_feature=edge_feature,
            directed=False,
            node_states=[0 for _ in range(cfg.gnn.layers_mp)],
            node_cells=[0 for _ in range(cfg.gnn.layers_mp)],
            node_list=node_indices
            )
        )

    return g_all

register_loader('timestep_dataset', load_generic_dataset)