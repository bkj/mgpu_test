import sys
import argparse
import numpy as np
from scipy.io import mmread

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, required=True)
    args = parser.parse_args()
    
    args.outpath = args.inpath.replace('.mmio', '.bin').replace('.mtx', '.bin')
    
    return args

args = parse_args()

print(f"reading: {args.inpath}", file=sys.stderr)
adj = mmread(args.inpath).tocsr()

sel = np.random.permutation(adj.shape[0])
adj = adj[sel][:,sel]

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