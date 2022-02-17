"""
Temporal transaction graph generator for PyG-Temporal
"""
import os
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


def get_temporal_dir_path(data_dir, freq):
    return os.path.join(data_dir, "freq_{}".format(freq))


def get_label_csv_path(data_dir, index):
    return os.path.join(data_dir, "label_{}.csv".format(index))


def get_feat_csv_path(data_dir, index):
    return os.path.join(data_dir, "feat_{}.csv".format(index))


def get_edge_csv_path(data_dir, index):
    return os.path.join(data_dir, "edge_{}.csv".format(index))


class DatasetLoader(object):
    
    @staticmethod
    def load_vertex_labels(label_csv_file):
        targets = np.loadtxt(label_csv_file).astype(int)
        return targets
    
    @staticmethod
    def load_edge_list(edge_csv_file):
        edge_index = np.loadtxt(edge_csv_file, delimiter=",").astype(int).T
        edge_weights = np.ones(len(edge_index))
        return edge_index, edge_weights
    
    @staticmethod
    def load_vertex_feats(feat_csv):
        arr = np.loadtxt(feat_csv, delimiter=",").astype(float)
        norm = (arr - arr.min(axis=0)) / arr.ptp(axis=0)  # Normalize
        return norm
    
    def __init__(self):
        self.num_nodes = None
        self.num_features = None
        self.num_snapshots = None
        self.edges = None  # List of 2D numpy arrays [src, dst]
        self.edge_weights = None  # List of numpy arrays
        self.features = None  # List of vertex features (numpy 2D array)
        self.targets = None  # List of vertex labels (numpy array)
    
    def _input_file_missing(self, input_file_path, t):
        if not os.path.isfile(input_file_path):
            print("File {} not found. Adjust max snapshots to {}".format(input_file_path, t))
            self.num_snapshots = t
            return True
        else:
            return False
    
    def _get_edges(self, data_dir):
        self.edges = list()
        self.edge_weights = list()
        for t in range(self.num_snapshots):
            edge_csv_path = get_edge_csv_path(data_dir, t)
            print("Loading edges:", edge_csv_path)
            if self._input_file_missing(edge_csv_path, t):
                return
            ei, ew = self.load_edge_list(edge_csv_path)
            self.edges.append(ei)
            self.edge_weights.append(ew)
    
    def _get_vertices(self, data_dir):
        self.targets = list()
        self.features = list()
        for t in range(self.num_snapshots):
            label_csv_path = get_label_csv_path(data_dir, t)
            feat_csv_path = get_feat_csv_path(data_dir, t)
            print("Loading vertices:", label_csv_path, feat_csv_path)
            for input_csv_path in [label_csv_path, feat_csv_path]:
                if self._input_file_missing(input_csv_path, t):
                    return
            targets = self.load_vertex_labels(label_csv_path)
            feats = self.load_vertex_feats(feat_csv_path)
            self.targets.append(targets)
            self.features.append(feats)
            num_nodes, num_feats = feats.shape
            self.num_nodes = self.num_nodes or num_nodes
            self.num_features = self.num_features or num_feats
    
    def get_dataset(self, data_dir, max_snapshots):
        self.num_snapshots = max_snapshots
        self._get_edges(data_dir)
        self._get_vertices(data_dir)
        dataset = DynamicGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets)
        return dataset
