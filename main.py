import numpy as np
import argparse
import networkx as nx
import time
import os
import sys
import pickle
from scipy import sparse  # Ensure this import is present

import cProfile, pstats

from utils import split_embeddings, split_adj, threshold_alignment_matrix, score_alignment_matrix, normalized_overlap
from refina import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run RefiNA.")
    parser.add_argument('--input', nargs='?', default='data/arenas_combined_edges.txt', help="Edgelist of combined input graph")
    parser.add_argument('--init-align', nargs='?', default='initial_solutions/arenas/regal.npz', help='Initial alignment matrix path')
    parser.add_argument('--n-iter', type=int, default=100, help='Maximum #iter for RefiNA. Default is 100.') #dimensions of other kinds of embeddings
    parser.add_argument('--n-update', type=int, default=-1, help='How many possible updates per node. Default is -1, or dense refinement.  Positive value uses sparse refinement')  
    parser.add_argument('--token-match', type=float, default = -1, help = "Token match score for each node.  Default of -1 sets it to reciprocal of largest graph #nodes rounded up to smallest power of 10")
    
    return parser.parse_args()

def main(args):
    # Read combined edgelist
    nx_graph = nx.read_edgelist(args.input, nodetype=int, comments="%")
    # Read initial benchmark alignment matrix
    adj = nx.adjacency_matrix(nx_graph, nodelist=range(nx_graph.number_of_nodes()))
    split_idx = 1124 if "species" in args.input else None  # for species dataset, networks are of different sizes and the first has 1124 nodes
    adj1, adj2 = split_adj(adj, split_idx)

    # Read in ground-truth node alignments, if available (if not available, true_alignments will be None)
    # RefiNA is unsupervised and does not need ground truth to run. This is just for evaluation

    # Assume true node alignments, if available, are given as a dictionary of node IDs. 
    # Follows REGAL implementation: https://github.com/GemsLab/REGAL
    true_alignments_fname = args.input.split("_")[0] + "_edges-mapping-permutation.txt" 
    true_alignments = None
    if os.path.exists(true_alignments_fname):
        with open(true_alignments_fname, "rb") as true_alignments_file:
            try:
                true_alignments = pickle.load(true_alignments_file)
            except:  # python3
                true_alignments = pickle.load(true_alignments_file, encoding="latin1")

    # All baselines besides REGAL variants are dense matrices so we can just load them in as a dense format
    try:
        init_alignment_matrix = sparse.load_npz(args.init_align)
    except:
        init_alignment_matrix = np.load(args.init_align)

    if args.n_update <= 0:  # dense operations
        if sparse.issparse(init_alignment_matrix):
            init_alignment_matrix = init_alignment_matrix.toarray()
        if sparse.issparse(adj1):
            adj1 = adj1.toarray()
        if sparse.issparse(adj2):
            adj2 = adj2.toarray()
    else:  # sparse operations
        if not sparse.issparse(init_alignment_matrix):
            init_alignment_matrix = sparse.csr_matrix(init_alignment_matrix)  # make sparse alignments
        if not sparse.issparse(adj1):
            adj1 = sparse.csr_matrix(adj1)
        if not sparse.issparse(adj2):
            adj2 = sparse.csr_matrix(adj2)

    # Keep top 1 alignment, i.e., treat initial solution as "hard" binary alignments
    # RefiNA can in principle refine "soft" alignments, but our paper only considered "hard" base alignments (which all base NA methods can produce)
    init_alignment_matrix = threshold_alignment_matrix(init_alignment_matrix, topk=1)
    # After splitting adj1 and adj2
    rows_for_adj1 = adj1.shape[0]  # e.g., 18524
    cols_for_adj2 = adj2.shape[0]  # e.g., 18523
    
    # Slice the alignment matrix from the loaded init_alignment_matrix
    # Ensure that init_alignment_matrix is defined in this scope
    sliced_alignment_matrix = init_alignment_matrix[:rows_for_adj1, rows_for_adj1:rows_for_adj1+cols_for_adj2]
    print(f"Sliced alignment matrix shape: {sliced_alignment_matrix.shape}")
    
    # Now call normalized_overlap with the sliced alignment matrix
    nov_score, lccc_score = normalized_overlap(adj1, adj2, sliced_alignment_matrix)
    rint("Initial normalized overlap %.5f%% and LCCC edge score %d" % (100*nov_score, lccc_score))
    if true_alignments is None:
        print("No ground truth alignments. Computing normalized overlap")
        # Use the refined alignment matrix or the correct sliced version here if needed
        nov_score, lccc_score = normalized_overlap(adj1, adj2, sliced_alignment_matrix)
        print("Initial normalized overlap %.5f%% and LCCC edge score %d" % (100*nov_score, lccc_score))		

    # Refine solution
    alignment_matrix = refina(init_alignment_matrix, adj1, adj2, args, true_alignments=true_alignments)
    # Save the refined alignment matrix as a sparse matrix
    refined_alignment_sparse_path = os.path.join(os.path.dirname(args.init_align), "refined_alignment_matrix.npz")
    if not sparse.isspmatrix_csr(alignment_matrix):
        alignment_matrix = alignment_matrix.tocsr()
    sparse.save_npz(refined_alignment_sparse_path, alignment_matrix)
    print(f"Refined alignment matrix saved to '{refined_alignment_sparse_path}'.")

    # Score final alignment result
    print("Refined alignment results:")
    if true_alignments is not None:
        score, _ = score_alignment_matrix(alignment_matrix, topk=1, true_alignments=true_alignments)
        print("Top 1 accuracy: %.5f" % score)
    else:
        print("No ground truth alignments.  Computing normalized overlap")
        if alignment_matrix.shape != (rows_for_adj1, cols_for_adj2):
            # Slice or adjust alignment_matrix accordingly
            alignment_matrix = alignment_matrix[:rows_for_adj1, :cols_for_adj2]
        nov_score, lccc_score = normalized_overlap(adj1, adj2, alignment_matrix)
        print("Normalized overlap %.5f%% and LCCC edge score %d" % (100*nov_score, lccc_score))

    mnc = score_MNC(alignment_matrix, adj1, adj2)
    mnc_output_path = os.path.join(os.path.dirname(args.init_align), "mnc_score.txt")
    # Save the mnc value to a text file
    with open(mnc_output_path, 'w') as f:
        f.write(f"MNC: {mnc:.3f}\n")

    print(f"MNC score saved to '{mnc_output_path}'.")
    print("MNC: %.3f" % mnc)

if __name__ == "__main__":
    args = parse_args()
    profile = cProfile.Profile()
    profile.run("main(args)")
    st = pstats.Stats(profile)
    st.sort_stats("cumtime")
    st.print_stats(0.1)
