import math
import warnings
import numpy as np
from numba import njit
from typing import Callable, Tuple
from .utils import _fast_shuffle, _soft_threshold, _mcp_threshold

L_FACTOR = 0.25
ETA_DIVISOR = 3.0

# Proximal operators
@njit(fastmath=True, cache=True)
def _prox_mcp(weight: float, step_size: float, lmbda: float) -> float:
    return _mcp_threshold(weight, lmbda, step_size)

@njit(fastmath=True, cache=True)
def _prox_lasso(weight: float, step_size: float, lmbda: float) -> float:
    return _soft_threshold(weight, step_size * lmbda)

@njit(fastmath=True, cache=True)
def _prox_ridge(weight: float, step_size: float, lmbda: float) -> float:
    return weight / (1.0 + 2.0 * step_size * lmbda)

@njit(fastmath=True, cache=True)
def _prox_none(weight: float, step_size: float, lmbda: float) -> float:
    return weight

# SAGA solver
@njit(fastmath=True, cache=True)
def _saga_binary(
        X: np.ndarray, y: np.ndarray, lambdas: np.ndarray, max_iter: int,
        tol: float, prox_op: Callable[[float, float, float], float],
) -> Tuple[np.ndarray, int]:
    
    n, d = X.shape
    
    w = np.zeros(d)
    previous_w = np.zeros(d)

    samples_seen = 1

    gradient_weights_memory = np.zeros(n)
    avg_gradient = np.zeros(d)
    permutation = np.arange(n, dtype=np.int32)

    L = L_FACTOR * np.max(np.sum(X * X, axis=1))
    step_size = 1.0 / (ETA_DIVISOR * L)

    for epoch in range(1, max_iter + 1):
        _fast_shuffle(permutation)
        previous_w[:] = w[:]

        for i in permutation:
            xi, yi = X[i], y[i]

            # Logistic prediction
            logit = 0.0
            for j in range(d):
                logit += w[j] * xi[j]
            prob = 1.0 / (1.0 + math.exp(-logit))

            # Gradient update
            gradient_weight = prob - yi
            delta_grad = gradient_weight - gradient_weights_memory[i]
            gradient_weights_memory[i] = gradient_weight

            for j in range(d):
                update = delta_grad * xi[j]
                w[j] -= step_size * (update + avg_gradient[j])
                if epoch == 1:
                    avg_gradient[j] += (update - avg_gradient[j]) / samples_seen
                else:
                    avg_gradient[j] += update / n
                w[j] = prox_op(w[j], step_size, lambdas[j])

            if epoch == 1:
                samples_seen += 1
        
        converged = True

        for j in range(d):
            if abs(w[j] - previous_w[j]) > tol * abs(w[j]):
                converged = False
                break
        
        if converged:
            return w, epoch
        
    return w, epoch

# Dictionary mapping
SCALAR_PROX_MAP = {
    "mcp": _prox_mcp,
    "lasso": _prox_lasso,
    "ridge": _prox_ridge,
    None: _prox_none,
}

# Public API functions
def fit_affine_scaling(
        X: np.ndarray, y: np.ndarray, penalty: str, reg_intercept: float, max_iter: int, tol: float
    ) -> np.ndarray:
    """Fit affine scaling."""
    lambdas = np.array([reg_intercept, 0.0])

    try:
        prox_op = SCALAR_PROX_MAP[penalty]
    except KeyError:
        raise ValueError(f"Unknown penalty: {penalty}")

    w, last_iter = _saga_binary(X, y, lambdas, max_iter, tol, prox_op)
    if last_iter == max_iter:
        warnings.warn("SAGA reached max_iter without convergence", RuntimeWarning)
    return w

def fit_quadratic_scaling(
    X: np.ndarray, y: np.ndarray, penalty: str, reg_intercept: float, reg_quadratic: float, max_iter: int, tol: float
    ) -> np.ndarray:
    """Fit quadratic scaling."""
    lambdas = np.array([reg_intercept, 0.0, reg_quadratic])

    try:
        prox_op = SCALAR_PROX_MAP[penalty]
    except KeyError:
        raise ValueError(f"Unknown penalty: {penalty}")

    w, last_iter = _saga_binary(X, y, lambdas, max_iter, tol, prox_op)
    if last_iter == max_iter:
        warnings.warn("SAGA reached max_iter without convergence", RuntimeWarning)
    return w

def warm_up_affine_scaling():
    """Warm up affine scaling functions."""
    X = np.random.randn(10, 2).astype(np.float32)
    y = np.random.randint(0, 2, size=10)
    for penalty in [None, 'mcp', 'lasso', 'ridge']:
        _saga_binary(X, y, np.array([0.1, 0.0]), 10, 1e-4, SCALAR_PROX_MAP[penalty])

def warm_up_quadratic_scaling():
    """Warm up quadratic scaling functions."""
    X = np.random.randn(10, 3).astype(np.float32)
    y = np.random.randint(0, 2, size=10)
    for penalty in [None, 'mcp', 'lasso', 'ridge']:
        _saga_binary(X, y, np.array([0.1, 0.0, 0.1]), 10, 1e-4, SCALAR_PROX_MAP[penalty])
