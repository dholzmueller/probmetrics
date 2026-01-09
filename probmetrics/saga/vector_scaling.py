import math
import warnings
import numpy as np
from numba import njit
from typing import Callable, Tuple
from .utils import _fast_shuffle, _soft_threshold, _mcp_threshold, _mcp_penalty

L_FACTOR = 0.25
ETA_DIVISOR = 3.0

# Logit computation
@njit(fastmath=True, cache=True)
def _compute_logits(
    xi: np.ndarray, v: np.ndarray, b: np.ndarray, k: int,
    logits: np.ndarray, exp_logits: np.ndarray
) -> float:
    """
    Computes numerically stable exp(logits) for a single sample.
    This function modifies logits and exp_logits in-place.

    Returns:
        The sum of the exponentiated logits (sum_exp).
    """
    max_logit = -np.inf
    for j in range(k):
        logit = (1.0 + v[j]) * xi[j] + b[j]
        logits[j] = logit
        if logit > max_logit:
            max_logit = logit

    sum_exp = 0.0
    for j in range(k):
        exp_logit = math.exp(logits[j] - max_logit)
        exp_logits[j] = exp_logit
        sum_exp += exp_logit
    
    return sum_exp

# Proximal operators
@njit(fastmath=True, cache=True)
def _prox_mcp(vec: np.ndarray, k: int, step_size: float, lmbda: float) -> None:
    """Applies MCP proximal operator inplace on each component of a k-dimensional parameter vector.
    """
    for i in range(k):
        vec[i] = _mcp_threshold(vec[i], lmbda, step_size)

@njit(fastmath=True, cache=True)
def _prox_lasso(vec: np.ndarray, k: int, step_size: float, lmbda: float) -> None:
    """Applies LASSO proximal operator inplace on each component of a k-dimensional parameter vector.
    """
    thresh = step_size * lmbda
    for i in range(k):
        vec[i] = _soft_threshold(vec[i], thresh)

@njit(fastmath=True, cache=True)
def _prox_ridge(vec: np.ndarray, k: int, step_size: float, lmbda: float) -> None:
    """Applies ridge proximal operator inplace on each component of a k-dimensional parameter vector.
    """
    factor = 1.0 / (1.0 + 2.0 * step_size * lmbda)
    for j in range(k):
        vec[j] *= factor

@njit(fastmath=True, cache=True)
def _prox_none(vec: np.ndarray, k: int, step_size: float, lmbda: float) -> None:
    pass

# Stopping functions
@njit(fastmath=True, cache=True)
def _compute_primal_dual_ridge(
    X: np.ndarray, y: np.ndarray, v: np.ndarray, b: np.ndarray,
    reg_intercept: float, reg_diagonal: float
) -> Tuple[float, float]:
    """Computes primal and (approximate) dual objectives for our structured vector scaling convex problem with ridge regularization.
    """
    n, k = X.shape
    
    primal_loss = 0.0
    dual_loss = 0.0
    s_v = np.zeros(k)
    s_b = np.zeros(k)
    
    logits = np.zeros(k)
    exp_logits = np.zeros(k)

    for i in range(n):
        xi, yi = X[i], y[i]

        sum_exp = _compute_logits(
            xi, v, b, k, logits, exp_logits
        )
        
        for j in range(k):
            prob = exp_logits[j] / sum_exp
            alpha_ij = prob
            dual_loss -= prob * math.log(prob)
            if j == yi:
                alpha_ij -= 1.0
                primal_loss -= math.log(prob)
            s_v[j] += alpha_ij * xi[j]
            s_b[j] += alpha_ij

    primal_reg = reg_diagonal * np.dot(v, v) + reg_intercept * np.dot(b, b)
    primal_obj = (primal_loss / n) + primal_reg

    n_squared = float(n * n)
    dual_reg = (- (1.0 / (4.0 * reg_diagonal * n_squared)) * np.dot(s_v, s_v)
                - (1.0 / (4.0 * reg_intercept * n_squared)) * np.dot(s_b, s_b))

    dual_lin = np.sum(s_v)

    dual_obj = (dual_loss / n) + (dual_lin / n) + dual_reg
    
    return primal_obj, dual_obj

@njit(fastmath=True, cache=True)
def _compute_primal_dual_lasso(
    X: np.ndarray, y: np.ndarray, v: np.ndarray, b: np.ndarray,
    reg_intercept: float, reg_diagonal: float
) -> Tuple[float, float]:
    """Computes primal and (approximate) dual objectives for our structured vector scaling convex problem with LASSO regularization.
    """
    n, k = X.shape
    
    primal_loss = 0.0
    dual_loss = 0.0
    dual_lin = 0.0
    
    logits = np.zeros(k)
    exp_logits = np.zeros(k)

    for i in range(n):
        xi, yi = X[i], y[i]

        sum_exp = _compute_logits(
            xi, v, b, k, logits, exp_logits
        )
        
        for j in range(k):
            prob = exp_logits[j] / sum_exp
            alpha_ij = prob
            dual_loss -= prob * math.log(prob)
            if j == yi:
                alpha_ij -= 1.0
                primal_loss -= math.log(prob)
            dual_lin += alpha_ij * xi[j]

    primal_reg = reg_diagonal * np.sum(np.abs(v)) + reg_intercept * np.sum(np.abs(b))
    primal_obj = (primal_loss / n) + primal_reg

    dual_obj = (dual_loss / n) + (dual_lin / n)

    return primal_obj, dual_obj

@njit(fastmath=True, cache=True)
def _compute_primal_dual_none(
    X: np.ndarray, y: np.ndarray, v: np.ndarray, b: np.ndarray,
    reg_intercept: float, reg_diagonal: float
) -> Tuple[float, float]:
    """Computes primal objective for our structured vector scaling convex problem without regularization.
    Since no regularization is applied, the dual cannot be evaluated. Only primal is used for stopping.
    """
    n, k = X.shape

    primal_loss = 0.0

    logits = np.zeros(k)
    exp_logits = np.zeros(k)

    for i in range(n):
        xi, yi = X[i], y[i]

        sum_exp = _compute_logits(
            xi, v, b, k, logits, exp_logits
        )
        
        primal_loss -= math.log(exp_logits[yi] / sum_exp)

    primal_obj = primal_loss / n

    return primal_obj, -np.inf

@njit(fastmath=True, cache=True)
def _compute_primal_dual_mcp(
    X: np.ndarray, y: np.ndarray, v: np.ndarray, b: np.ndarray,
    reg_intercept: float, reg_diagonal: float
) -> Tuple[float, float]:
    """Computes primal objective for our structured vector scaling convex problem with MCP regularization.
    With MCP penalty, the dual cannot be evaluated. Only primal is used for stopping.
    """
    n, k = X.shape
    
    primal_loss = 0.0
    
    logits = np.zeros(k)
    exp_logits = np.zeros(k)

    for i in range(n):
        xi, yi = X[i], y[i]

        sum_exp = _compute_logits(
            xi, v, b, k, logits, exp_logits
        )
        
        primal_loss -= math.log(exp_logits[yi] / sum_exp)

    primal_reg = 0.0
    for j in range(k):
        primal_reg += _mcp_penalty(v[j], reg_diagonal)
        primal_reg += _mcp_penalty(b[j], reg_intercept)

    primal_obj = (primal_loss / n) + primal_reg

    return primal_obj, -np.inf

# SAGA solver
@njit(fastmath=True, cache=True)
def _saga_vector_scaling(
    X: np.ndarray, y: np.ndarray, reg_intercept: float, reg_diagonal: float,
    max_iter: int, tol: float, prox_op: Callable[[np.ndarray, int, float, float], None], stop_op: Callable[..., float]
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """Solves our structured vector scaling convex optimization problem using SAGA.

    Args:
        X (np.ndarray): un-calibrated logits.
        y (np.ndarray): un-calibrated labels (not one-hot encoded).
        reg_intercept (float): regularization strenght applied to intercept parameters.
        reg_diagonal (float): regularization strenght applied to diagonal parameters.
        max_iter (int): max number of dataset passes before SAGA is stopped.
        tol (float): tolerance used for stopping.
            Stopping is based on relative difference between primal and (approximate) dual objectives when available.
            Otherwise relative difference between consecutive primal objectives is used.
        prox_op (Callable[[np.ndarray, int, float, float], None]): proximal operator used to regularize.
        stop_op (Callable[..., float]): function that computes the primal and dual objectives for stopping.

    Returns:
        Tuple[float, np.ndarray, np.ndarray, int]: global scaling parameter a, vector scaling parameter v, intercept parameter b, number of epochs performed.
    """
    n, k = X.shape
    
    v = np.zeros(k)
    b = np.zeros(k)

    previous_objective = np.inf
    samples_seen = 1

    gradient_weights_memory = np.zeros((n, k))
    permutation = np.arange(n, dtype=np.int32)
    avg_gradient_v = np.zeros(k)
    avg_gradient_b = np.zeros(k)
    
    logits = np.zeros(k)
    exp_logits = np.zeros(k)
    delta_grad = np.zeros(k)

    L = L_FACTOR * np.max(np.sum(X * X, axis=1))
    step_size = 1.0 / (ETA_DIVISOR * L)

    for epoch in range(1, max_iter + 1):
        _fast_shuffle(permutation)
        
        for i in permutation:
            xi, yi = X[i], y[i]

            sum_exp = _compute_logits(
                xi, v, b, k, logits, exp_logits
            )

            for j in range(k):
                prob = exp_logits[j] / sum_exp
                gradient_weight = prob - (1.0 if j == yi else 0.0)

                delta_grad[j] = gradient_weight - gradient_weights_memory[i, j]
                gradient_weights_memory[i, j] = gradient_weight
                
                update_v = delta_grad[j] * xi[j]
                v[j] -= step_size * (update_v + avg_gradient_v[j])
                if epoch == 1:
                    avg_gradient_v[j] += (update_v - avg_gradient_v[j]) / samples_seen
                else:
                    avg_gradient_v[j] += update_v / n

                update_b = delta_grad[j]
                b[j] -= step_size * (update_b + avg_gradient_b[j])
                if epoch == 1:
                    avg_gradient_b[j] += (update_b - avg_gradient_b[j]) / samples_seen
                else:
                    avg_gradient_b[j] += update_b / n

            if epoch == 1:
                samples_seen += 1

            prox_op(v, k, step_size, reg_diagonal)
            prox_op(b, k, step_size, reg_intercept)

        primal_obj, dual_obj = stop_op(X, y, v, b, reg_intercept, reg_diagonal)
        if dual_obj == -np.inf:
            gap = abs(primal_obj - previous_objective)
            if gap <= tol * max(1.0, abs(primal_obj)):
                return v, b, epoch
            previous_objective = primal_obj
        else:
            gap = abs(primal_obj - dual_obj)
            if gap <= tol * max(1.0, abs(primal_obj)):
                return v, b, epoch

    return v, b, max_iter

# Dictionary mappings
VECTOR_PROX_MAP = {
    "mcp": _prox_mcp,
    "lasso": _prox_lasso,
    "ridge": _prox_ridge,
    None: _prox_none,
}

VECTOR_STOP_MAP = {
    "mcp": _compute_primal_dual_mcp,
    "lasso": _compute_primal_dual_lasso,
    "ridge": _compute_primal_dual_ridge,
    None: _compute_primal_dual_none,
}

# Public API functions
def fit_vector_scaling(
        X: np.ndarray, y: np.ndarray, penalty: str, reg_diagonal: float,
        reg_intercept: float, max_iter: int, tol: float
) -> np.ndarray:
    """Public API to fit vector scaling using SAGA."""
    try:
        prox_op = VECTOR_PROX_MAP[penalty]
        stop_op = VECTOR_STOP_MAP[penalty]
    except KeyError:
        raise ValueError(f"Unknown penalty: {penalty}")

    v_delta, b, last_iter = _saga_vector_scaling(
        X, y, reg_intercept, reg_diagonal,
        max_iter, tol, prox_op, stop_op
    )
    if last_iter == max_iter:
        warnings.warn("SAGA reached max_iter without convergence", RuntimeWarning)
    
    return 1.0 + v_delta, b

def warm_up_vector_scaling():
    """Public API to warm up vector scaling."""
    X = np.random.randn(10, 3).astype(np.float32)
    y = np.random.randint(0, 3, size=10)
    for penalty in [None, 'mcp', 'lasso', 'ridge']:
        _saga_vector_scaling(X, y, 3.0/10.0, 3.0/10.0, 10, 1e-5, VECTOR_PROX_MAP[penalty], VECTOR_STOP_MAP[penalty])
