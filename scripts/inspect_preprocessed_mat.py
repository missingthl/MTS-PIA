
import scipy.io
import numpy as np
import os

path = "data/SEED/SEED_EEG/Preprocessed_EEG/1_20131027.mat"
if not os.path.exists(path):
    print(f"Not found: {path}")
    exit(1)

mat = scipy.io.loadmat(path)
print(f"Keys: {list(mat.keys())}")

# Check one key
for k in mat.keys():
    if k.startswith("__"): continue
    data = mat[k]
    print(f"Key {k}: Shape {data.shape}, Type {data.dtype}")
    print(f"Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    print(f"Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
    break
