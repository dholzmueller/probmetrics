import numpy as np
from numba import njit

MCP_GAMMA = 3.0

@njit(fastmath=True, cache=True)
def _fast_shuffle(arr: np.ndarray) -> None:
    """In-place accelerated array shuffling."""
    for i in range(arr.shape[0] - 1, 0, -1):
        j = np.random.randint(i + 1)
        arr[i], arr[j] = arr[j], arr[i]

@njit(fastmath=True, inline="always", cache=True)
def _soft_threshold(val: float, thresh: float) -> float:
    """Soft-thresholding of val at level thresh."""
    if val > thresh:
        return val - thresh
    elif val < -thresh:
        return val + thresh
    else:
        return 0.0

@njit(fastmath=True, inline="always", cache=True)
def _mcp_threshold(val: float, lmbda: float, step_size: float, gamma: float = MCP_GAMMA) -> float:
    """MCP-thresholding of val with regularization strength lambda and non-convexity parameter gamma."""
    thresh = lmbda * step_size
    abs_val = abs(val)

    if abs_val <= thresh:
        return 0.0

    gamma_lmbda = gamma * lmbda
    if abs_val > gamma_lmbda:
        return val

    div = 1.0 - step_size / gamma
    if val > thresh:
        return (val - thresh) / div
    else: # val < -thresh
        return (val + thresh) / div

@njit(fastmath=True, cache=True)
def _mcp_penalty(val: float, lmbda: float, gamma: float = MCP_GAMMA) -> float:
    """Computes the value of the MCP penalty."""
    abs_val = abs(val)
    if abs_val <= gamma * lmbda:
        return lmbda * abs_val - (abs_val**2) / (2.0 * gamma)
    else:
        return 0.5 * gamma * lmbda**2
