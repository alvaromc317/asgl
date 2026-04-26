from typing import Union
import numpy as np
from scipy import sparse

# Custom types
ArrayOrSparse = Union[np.ndarray, sparse.spmatrix]

# Define constants for penalization types
INDIV_NONADAPTIVE = ["lasso", "ridge", "sgl"]
INDIV_ADAPTIVE = ["alasso", "aridge", "asgl"]
GROUP_NONADAPTIVE = ["gl", "sgl"]
GROUP_ADAPTIVE = ["agl", "asgl"]
ALL_PENALTIES = INDIV_NONADAPTIVE + INDIV_ADAPTIVE + GROUP_ADAPTIVE + GROUP_NONADAPTIVE
ALLOWED_MODELS = ["lm", "qr", "logit"]
ALLOWED_WEIGHT_TECHNIQUES = {
  "pca_1",
  "pca_pct",
  "pls_1",
  "pls_pct",
  "sparse_pca",
  "unpenalized",
  "lasso",
  "ridge",
}
ALLOWED_CANON_BACKENDS = {
  "CPP",
  "SCIPY",
  "COO",
}
