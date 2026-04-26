from typing import Tuple, Dict
import numpy as np


def _get_group_info(
  group_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
  """
  Efficiently computes group sizes and indices for each group.
  """
  argsort_indices = np.argsort(group_index, kind="mergesort")
  sorted_group_index = group_index[argsort_indices]
  unique_groups, group_starts, group_counts = np.unique(
    sorted_group_index, return_index=True, return_counts=True
  )
  indices_per_group = {
    g: argsort_indices[start : start + count]
    for g, start, count in zip(unique_groups, group_starts, group_counts)
  }
  return unique_groups, group_counts, indices_per_group
