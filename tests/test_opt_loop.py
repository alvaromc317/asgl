import time
import numpy as np
import scipy.sparse as sp

group_index = np.repeat(np.arange(40), 5)
unique_groups, group_starts, group_counts = np.unique(group_index, return_index=True, return_counts=True)

start = time.time()
for _ in range(10000):
    argsort_indices = np.argsort(group_index, kind='mergesort')
    sorted_group_index = group_index[argsort_indices]
    unique_groups, group_starts, group_counts = np.unique(sorted_group_index, return_index=True, return_counts=True)
    indices_per_group = {
        g: argsort_indices[start : start + count]
        for g, start, count in zip(unique_groups, group_starts, group_counts)
    }
print(f"Original: {time.time() - start:.4f}s")

start = time.time()
for _ in range(10000):
    unique_groups, group_counts = np.unique(group_index, return_counts=True)
    argsort_indices = np.argsort(group_index, kind='mergesort')
    sorted_group_index = group_index[argsort_indices]
    group_starts = np.searchsorted(sorted_group_index, unique_groups)

    indices_per_group = {
        g: argsort_indices[start : start + count]
        for g, start, count in zip(unique_groups, group_starts, group_counts)
    }
print(f"New searchsorted: {time.time() - start:.4f}s")

start = time.time()
for _ in range(10000):
    argsort_indices = np.argsort(group_index, kind='mergesort')
    unique_groups, inverse_indices, group_counts = np.unique(group_index, return_inverse=True, return_counts=True)
    # Actually just split argsort_indices by group counts
    group_starts = np.cumsum(group_counts)[:-1]
    splits = np.split(argsort_indices, group_starts)
    indices_per_group = dict(zip(unique_groups, splits))
print(f"Split argsort: {time.time() - start:.4f}s")
