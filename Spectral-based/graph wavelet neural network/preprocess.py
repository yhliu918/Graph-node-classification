import numpy as np
import networkx as nx
import scipy.sparse as sp
import math
import pickle as pkl
from sklearn.preprocessing import normalize
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.special import iv
from scipy.integrate import quad
import sys

def parse_index_file(filename):
	"""Parse index file."""
	index = []
	for line in open(filename):
		index.append(int(line.strip()))
	return index


def sample_mask(idx, l):
	"""Create mask."""
	mask = np.zeros(l)
	mask[idx] = 1
	return np.array(mask, dtype=np.bool)

def load_data(dataset_str):
	"""
	ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
	"""
	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
	objects = []
	for i in range(len(names)):
		with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
			if sys.version_info > (3, 0):
				objects.append(pkl.load(f, encoding='latin1'))
				"""
				encoding 告诉 pickle 如何解码 Python 2 存储的 8 位字符串实默认分别为 'ASCII'
				encoding 参数可置为 'bytes' 来将这些 8 位字符串实例读取为字节对象。
				读取 NumPy array 和 Python 2 存储的 datetime、date 和 time 实例时，请使用 encoding='latin1'。
				"""
			else:
				objects.append(pkl.load(f))

	x, y, tx, ty, allx, ally, graph = tuple(objects)
	test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
	test_idx_range = np.sort(test_idx_reorder)

	features = sp.vstack((allx, tx)).tolil()
	features[test_idx_reorder, :] = features[test_idx_range, :]
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

	labels = np.vstack((ally, ty))
	labels[test_idx_reorder, :] = labels[test_idx_range, :]

	classes_num = labels.shape[1]

	isolated_list = [i for i in range(labels.shape[0]) if np.all(labels[i] == 0)]
	if isolated_list:
		print(f"Warning: Dataset '{dataset_str}' contains {len(isolated_list)} isolated nodes")

	labels = np.array([np.argmax(row) for row in labels], dtype=np.long)

	idx_test = test_idx_range.tolist()
	idx_train = range(len(y))
	idx_val = range(len(y), len(y)+500)

	train_mask = sample_mask(idx_train, labels.shape[0])
	val_mask = sample_mask(idx_val, labels.shape[0])
	test_mask = sample_mask(idx_test, labels.shape[0])

	y_train = labels[train_mask]
	y_val = labels[val_mask]
	y_test = labels[test_mask]

	return adj, features, y_train, y_val, y_test, classes_num, train_mask, val_mask, test_mask

def preprocess_features(features):
    #Row-normalize feature matrix and convert to tuple representation
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv,0)
    features = r_mat_inv.dot(features)
    return features