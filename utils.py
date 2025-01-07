import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KDTree, NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import normalize
import networkx as nx

from collections import defaultdict
import os

'''
=======================================
==================Data handling==================
=======================================
'''

# Split embeddings in two
def split_embeddings(combined_embed, split_index=None, increasing_size=True):
    if split_index is None:
        split_index = int(combined_embed.shape[0] / 2)  # default: assume graphs are same size
    embed1 = combined_embed[:split_index]
    embed2 = combined_embed[split_index:]

    # Align larger graph to smaller one
    if increasing_size and embed1.shape[0] < embed2.shape[0]:
        tmp = embed1
        embed1 = embed2
        embed2 = tmp

    return embed1, embed2

# Split adjacency matrix in two
def split_adj(combined_adj, split_index=None, increasing_size=True):
    total_nodes = combined_adj.shape[0]
    if split_index is None:
        split_index = total_nodes // 2  # default: assume graphs are same size
    adj1 = combined_adj[:split_index, :split_index]
    adj2 = combined_adj[split_index:, split_index:]
    
    if sp.issparse(combined_adj):
        if combined_adj.getformat() != "csc":
            combined_adj = combined_adj.tocsc()
        adj1 = combined_adj[:, :split_index]
        adj2 = combined_adj[:, split_index:]  # select columns as csc because faster
        adj1 = adj1.tocsr()
        adj2 = adj2.tocsr()  # convert to CSR for fast row slicing
        adj1 = adj1[:split_index]
        adj2 = adj2[split_index:]
    else:
        adj1 = combined_adj[:split_index, :split_index]
        adj2 = combined_adj[split_index:, split_index:]
    
    # Align larger graph to smaller one
    if increasing_size and adj1.shape[0] < adj2.shape[0]:
        tmp = adj1
        adj1 = adj2
        adj2 = tmp

    return adj1, adj2

'''
=======================================
==================I/O==================
=======================================
'''
# Wrappers to save/load either sparse or dense matrix
def save_alignment_matrix(alignment_path, alignment_matrix, overwrite=True):
    alignment_fname = alignment_path
    if sp.issparse(alignment_matrix):
        if not alignment_fname.endswith(".npz"):
            alignment_fname = f"{alignment_fname}.npz"
    if overwrite or not os.path.exists(alignment_fname):
        sp.save_npz(alignment_fname, alignment_matrix)
    else:
        if not alignment_fname.endswith(".npy"):
            alignment_fname = f"{alignment_fname}.npy"
        if overwrite or not os.path.exists(alignment_fname):
            np.savetxt(alignment_fname, alignment_matrix)

def load_alignment_matrix(alignment_path):
    if not (alignment_path.endswith("npz") or alignment_path.endswith(".npy")):  # no extension given--see if either exists, looking for sparse one first
        if os.path.exists(f"{alignment_path}.npz"):
            alignment_path = f"{alignment_path}.npz"
        else:
            alignment_path = f"{alignment_path}.npy"
    if alignment_path.endswith(".npz"):  # sparse matrix
        alignment_matrix = sp.load_npz(alignment_path)
    else:  # dense matrix
        try:
            alignment_matrix = np.loadtxt(alignment_path)
        except:
            alignment_matrix = np.load(alignment_path)
    return alignment_matrix

'''
=======================================
==================Scoring==================
=======================================
'''
'''Score (soft correspondence) alignment matrix given true alignments'''
# Note: make sure alignment matrix, if dense is np.ndarray, not numpy matrix (which NumPy recommends not using anyway)
def score_alignment_matrix(alignment_matrix, topk=1, topk_score_weighted=False, true_alignments=None):
    n_nodes = alignment_matrix.shape[0]
    correct_nodes = defaultdict(list)

    alignment_score = defaultdict(int)
    if sp.issparse(alignment_matrix):
        if alignment_matrix.shape[0] > 2e4:
            return score_sparse_alignment_matrix(alignment_matrix, topk, topk_score_weighted, true_alignments)
        else:  # convert to dense if small enough
            alignment_matrix = alignment_matrix.toarray()

    if not sp.issparse(alignment_matrix):
        sorted_indices = np.argsort(alignment_matrix)
    for node_index in range(n_nodes):
        target_alignment = node_index  # default: assume identity mapping, and the node should be aligned to itself
        if true_alignments is not None:  # if we have true alignments (which we require), use those for each node
            target_alignment = int(true_alignments[node_index])
        if sp.issparse(alignment_matrix):
            row, possible_alignments, possible_values = sp.find(alignment_matrix[node_index])
            node_sorted_indices = possible_alignments[possible_values.argsort()]
        else:
            node_sorted_indices = sorted_indices[node_index]
        node_sorted_indices = node_sorted_indices.T.ravel()
        
        if isinstance(topk, int):
            topk = [topk]
        for kval in topk:
            if target_alignment in node_sorted_indices[-kval:]:
                if topk_score_weighted:
                    alignment_score[kval] += 1.0 / (n_nodes - np.argwhere(sorted_indices[node_index] == target_alignment)[0])
                else:
                    alignment_score[kval] += 1
                correct_nodes[kval].append(node_index)
    for kval in topk:
        alignment_score[kval] /= float(n_nodes)  # normalize
    if len(topk) == 1:
        alignment_score = alignment_score[topk[0]]  # only wanted one score, so return just that one score

    return alignment_score, correct_nodes

def score_sparse_alignment_matrix(alignment_matrix, topk=1, topk_score_weighted=False, true_alignments=None):
    n_nodes = alignment_matrix.shape[0]
    correct_nodes = defaultdict(list)
    alignment_score = defaultdict(int)

    sparse_format = alignment_matrix.getformat()
    if sparse_format != "lil":
        alignment_matrix = alignment_matrix.tolil()

    for node_index in range(n_nodes):
        target_alignment = node_index  # default: assume identity mapping, and the node should be aligned to itself
        if true_alignments is not None:  # if we have true alignments (which we require), use those for each node
            target_alignment = int(true_alignments[node_index])
        
        sorted_indices = np.argsort(alignment_matrix.data[node_index])  # sorted indices nonzero values only
        node_sorted_indices = np.asarray(alignment_matrix.rows[node_index])[sorted_indices]  # sorted indices in the whole thing

        if isinstance(topk, int):
            topk = [topk]
        for kval in topk:
            if target_alignment in node_sorted_indices[-kval:]:
                if topk_score_weighted:
                    alignment_score[kval] += 1.0 / (n_nodes - np.argwhere(sorted_indices[node_index] == target_alignment)[0])
                else:
                    alignment_score[kval] += 1
                correct_nodes[kval].append(node_index)
    
    for k in alignment_score:
        alignment_score[k] /= float(n_nodes)
    if len(topk) == 1:
        alignment_score = alignment_score[topk[0]]  # we only wanted one score: return just this score instead of a dict of scores

    alignment_matrix = alignment_matrix.tocsr()
    return alignment_score, correct_nodes

def normalized_overlap(adj1, adj2, alignment_matrix, compute_lccc=True):
    alignment_matrix = threshold_alignment_matrix(alignment_matrix, topk=1)  # binarize, keep top 1 alignment
    print(f"Alignment matrix shape: {alignment_matrix.shape}")
    print(f"Adjacency matrix adj1 shape: {adj1.shape}")
    print(f"Adjacency matrix adj2 shape: {adj2.shape}")
    # Permute graph1 using discovered alignments
    if sp.issparse(adj1):
        alignment_matrix = sp.csr_matrix(alignment_matrix)  # so no weird things with sparse/dense multiplication
        adj1 = adj1.tocsr()
        adj2 = adj2.tocsr()  # just make sure we use the same sparse format

    map_adj1 = alignment_matrix.T.dot(adj1).dot(alignment_matrix)
    if sp.issparse(map_adj1):  # adj matrices are sparse and so is overlap matrices
        overlap_edges = map_adj1.multiply(adj2)
        n_overlap = overlap_edges.nnz
        max_edges = max(adj1.nnz, adj2.nnz)
    else:
        overlap_edges = map_adj1 * adj2
        n_overlap = np.count_nonzero(overlap_edges)
        max_edges = max(np.count_nonzero(adj1), np.count_nonzero(adj2))
    
    lccc_edges = -1
    if compute_lccc:
        overlap_nx = nx.from_scipy_sparse_matrix(sp.csr_matrix(overlap_edges))
        lccc = max(nx.connected_components(overlap_nx), key=len)  # NX 2.5
        lccc = overlap_nx.subgraph(lccc).copy()  # NX 2.5
        lccc_nodes = lccc.number_of_nodes()
        lccc_edges = lccc.number_of_edges()
        print(f"{lccc_nodes} nodes and {lccc_edges} edges in largest conserved connected component")
    nov = n_overlap / float(max_edges)
    return nov, lccc_edges


'''
=======================================
==================Thresholding/normalizing==================
=======================================
'''

# https://stackoverflow.com/questions/54984809/numpy-sort-each-row-and-retrieve-kth-element
def kth(dist, k):
    return np.sort(np.partition(dist, k-1, axis=1)[:, k-1])

# Keep only top k entries (topk = None keeps top 1 with ties)
def threshold_alignment_matrix(M, topk=None, keep_dist=False):
    '''slow, so use dense ops for smaller matrices'''
    sparse_input = sp.issparse(M)
    if sparse_input:
        if M.shape[0] > 20000:
            return threshold_alignment_matrix_sparse(M, topk, keep_dist)  # big matrix, use sparse format for memory reasons
        else:
            M = M.toarray()  # on smaller matrices, dense is fastest

    if topk is None or topk <= 0:  # top-1, 0-1
        row_maxes = M.max(axis=1).reshape(-1, 1)
        M[:] = np.where(M == row_maxes, 1, 0)  # keeps ties
        M[M < 1] = 0
        if sparse_input:
            M = sp.csr_matrix(M)
        return M
    else:  # selects one tie arbitrarily
        ind = np.argpartition(M, -topk)[:, -topk:]
        row_idx = np.arange(len(M)).reshape((len(M), 1)).repeat(topk, axis=1)  # n x k matrix of [1...n] repeated k times
        
        M_thres = np.zeros(M.shape)
        if keep_dist:
            vals = M[row_idx, ind]
            M_thres[row_idx, ind] = vals
        else:
            M_thres[row_idx, ind] = 1
        if sparse_input:
            M_thres = sp.csr_matrix(M_thres)
        return M_thres

def threshold_alignment_matrix_sparse(M, topk=1, keep_dist=False):
    if topk is None:
        topk = 1  # Default to top-1 alignment
    
    if topk == 1:  # we can find just the max elements per row easier
        max_indices = np.ravel(np.asarray(M.argmax(axis=1)))
        max_vals = np.ravel(M.max(axis=1).toarray())  # probably redundant but fast
        # max indices are columns of maximum values; corresponding row entries are just 0 to M.shape[1] - 1
        return sp.csr_matrix((max_vals, (np.arange(len(max_indices)), max_indices)), shape=M.shape)

    # https://stackoverflow.com/questions/36135927/get-top-n-items-of-every-row-in-a-scipy-sparse-matrix
    print(f"thresholding sparse matrix of format {M.getformat()}...")
    if M.getformat() != "lil":
        print("converting to lil...")
        M = M.tolil()

    def max_n(row_data, row_indices, n):
        i = np.argpartition(row_data, -n)[-n:]
        top_values = row_data[i]
        top_indices = row_indices[i]
        return top_values, top_indices, i

    for i in range(M.shape[0]):
        if len(M.data[i]) > topk:
            d, r = max_n(np.array(M.data[i]), np.array(M.rows[i]), topk)[:2]
            if keep_dist:
                M.data[i] = d.tolist()
            else:
                M.data[i] = [1] * len(d)
            M.rows[i] = r.tolist()

    return M.tocsr()

def skp_alg(M, max_iter=1000, tol=1e-2):
    for i in range(max_iter):
        # Check for convergence
        max_thresh = 1 + tol
        min_thresh = 1 - tol
        converged = True
        row_sums = M.sum(axis=1)
        col_sums = M.sum(axis=0)
        if row_sums.min() < min_thresh or row_sums.max() > max_thresh or col_sums.min() < min_thresh or col_sums.max() > max_thresh:
            converged = False
        if converged:
            print(f"Converged to tolerance of {tol} after {i} iterations")
            return M

        # Row normalize and column normalize
        M = normalize(M, norm="l1", axis=1)
        M = normalize(M, norm="l1", axis=0)

    print(f"Max number of iterations {max_iter} met")
    return M


'''
=======================================
==================Aligning==================
=======================================
'''

'''Softmax normalization'''
def softmax(M, theta=1.0, axis=1):
    if sp.issparse(M):
        if M.getformat() != "csr":
            M = M.tocsr()  # convert to CSR if not already CSR
        exp = (M * theta).expm1()  # exponent - 1, so that zeros remain zero
        return normalize(exp, norm="l1", axis=1)
    else:
        exp = np.exp(M * theta)
        return exp / np.sum(exp, axis=axis)[:, None]

def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=10):
    kd_tree = KDTree(emb2, metric=distance_metric)
        
    row = np.array([])
    col = np.array([])
    data = np.array([])
    
    dist, ind = kd_tree.query(emb1, k=num_top)
    print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = sp.coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()
