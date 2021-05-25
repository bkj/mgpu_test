import sys
import argparse
import numpy as np
from scipy import sparse
from scipy.io import mmread

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='graph500-scale18-ef16_adj.mmio')
    parser.add_argument('--seed',   type=int, default=123)
    args = parser.parse_args()
    
    args.outpath = args.inpath.replace('.mmio', '.bin').replace('.mtx', '.bin')
    
    return args

from matplotlib import pyplot as plt
from rcode import *

def plot_degrees():
    degrees = adj @ np.ones(adj.shape[0])
    _ = plt.scatter(np.arange(len(degrees)), degrees, alpha=0.2)
    show_plot()

args = parse_args()
np.random.seed(args.seed)

print(f"reading: {args.inpath}", file=sys.stderr)
adj = mmread(args.inpath).tocsr()
adj.data[:] = 1

# drop zero degrees
degrees = adj @ np.ones(adj.shape[0])
sel = degrees > 0
adj = adj[sel][:,sel]

# permute
sel = np.random.permutation(adj.shape[0])
adj = adj[sel][:,sel]

# # truncate degree
# degrees = adj @ np.ones(adj.shape[0])
# sel = degrees < np.percentile(degrees, 99)
# adj = adj[sel][:,sel]

# drop zero degrees
degrees = adj @ np.ones(adj.shape[0])
sel = degrees > 0
adj = adj[sel][:,sel]

# interleave degrees
degrees = adj @ np.ones(adj.shape[0])
sel = np.argsort(degrees)

sel = np.hstack([
    np.random.permutation(sel[0::4]), 
    np.random.permutation(sel[1::4]), 
    np.random.permutation(sel[2::4]), 
    np.random.permutation(sel[3::4])
])
adj = adj[sel][:,sel]
adj.eliminate_zeros()
adj.sort_indices()

shape   = np.array(adj.shape).astype(np.int32)
nnz     = np.array([adj.nnz]).astype(np.int32)
indptr  = adj.indptr.astype(np.int32)
indices = adj.indices.astype(np.int32)
data    = adj.data.astype(np.float32)

print(f"writing: {args.outpath}", file=sys.stderr)
with open(args.outpath, 'wb') as f:
    _ = f.write(bytearray(shape))
    _ = f.write(bytearray(nnz))
    _ = f.write(bytearray(indptr))
    _ = f.write(bytearray(indices))
    _ = f.write(bytearray(data))
    f.flush()