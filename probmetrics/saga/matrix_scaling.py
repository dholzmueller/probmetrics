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
    xi: np.ndarray, W_delta: np.ndarray, k: int, k_plus_1: int,
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
        logit = xi[j]
        for l in range(k_plus_1):
            logit += W_delta[j, l] * xi[l]
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
def _prox_mcp(W: np.ndarray, k: int, step_size: float, reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float) -> None:
    """Inplace Minimax Concave Penalty (MCP) proximal operator for the weight matrix of our hierarchical Softmax regression.
    See "Nearly unbiased variable selection under minimax concave penalty" https://arxiv.org/abs/1002.4734
    """
    for j in range(k):
        W[j, k] = _mcp_threshold(W[j, k], reg_intercept, step_size) # intercepts
        W[j, j] = _mcp_threshold(W[j, j], reg_diagonal, step_size) # diagonal
        for l in range(j):
            W[j, l] = _mcp_threshold(W[j, l], reg_off_diagonal, step_size) # off-diagonal
            W[l, j] = _mcp_threshold(W[l, j], reg_off_diagonal, step_size) # off-diagonal

@njit(fastmath=True, cache=True)
def _prox_lasso(W: np.ndarray, k: int, step_size: float, reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float) -> None:
    """Inplace LASSO proximal operator for the weight matrix of our hierarchical Softmax regression."""
    thresh_intercept = step_size * reg_intercept
    thresh_diagonal = step_size * reg_diagonal
    thresh_off_diagonal = step_size * reg_off_diagonal

    for j in range(k):
        W[j, k] = _soft_threshold(W[j, k], thresh_intercept) # intercept
        W[j, j] = _soft_threshold(W[j, j], thresh_diagonal) # diagonal
        for l in range(j):
            W[j, l] = _soft_threshold(W[j, l], thresh_off_diagonal) # off-diagonal
            W[l, j] = _soft_threshold(W[l, j], thresh_off_diagonal) # off-diagonal

@njit(fastmath=True, cache=True)
def _prox_group_lasso(W: np.ndarray, k: int, step_size: float, reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float) -> None:
    """Inplace group-lasso proximal operator for the weight matrix of our hierarchical Softmax regression."""
    # Computing norms
    norm_intercept_sq = 0.0
    norm_diagonal_sq = 0.0
    norm_off_diagonal_sq = 0.0

    for j in range(k):
        norm_intercept_sq += W[j, k] ** 2
        norm_diagonal_sq += W[j, j] ** 2
        for l in range(j):
            norm_off_diagonal_sq += W[j, l] ** 2 + W[l, j] ** 2

    # Proximal step intercept
    if norm_intercept_sq > 0.0:
        shrinkage = max(0.0, 1.0 - step_size * reg_intercept / math.sqrt(norm_intercept_sq))
        for j in range(k):
            W[j, k] *= shrinkage

    # Proximal step diagonal
    if norm_diagonal_sq > 0.0:
        shrinkage = max(0.0, 1.0 - step_size * reg_diagonal / math.sqrt(norm_diagonal_sq))
        for j in range(k):
            W[j, j] *= shrinkage

    # Proximal step off-diagonal
    if norm_off_diagonal_sq > 0.0:
        shrinkage = max(0.0, 1.0 - step_size * reg_off_diagonal / math.sqrt(norm_off_diagonal_sq))
        for j in range(k):
            for l in range(j):
                W[j, l] *= shrinkage
                W[l, j] *= shrinkage

@njit(fastmath=True, cache=True)
def _prox_ridge(W: np.ndarray, k: int, step_size: float, reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float) -> None:
    """Inplace ridge proximal operator for the weight matrix of our hierarchical Softmax regression."""
    factor_intercept = 1.0 / (1.0 + 2.0 * step_size * reg_intercept)
    factor_diagonal = 1.0 / (1.0 + 2.0 * step_size * reg_diagonal)
    factor_off_diagonal = 1.0 / (1.0 + 2.0 * step_size * reg_off_diagonal)

    for j in range(k):
        W[j, k] *= factor_intercept # intercept
        W[j, j] *= factor_diagonal # diagonal
        for l in range(j):
            W[j, l] *= factor_off_diagonal # off-diagonal
            W[l, j] *= factor_off_diagonal # off-diagonal

@njit(fastmath=True, cache=True)
def _prox_none(W: np.ndarray, k: int, step_size: float, reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float) -> None:
    pass

# Stopping functions
@njit(fastmath=True, cache=True)
def _compute_primal_dual_ridge(
    X: np.ndarray, y: np.ndarray, W_delta: np.ndarray,
    reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float
) -> Tuple[float, float]:
    """Computes primal and (approximate) dual objectives for our structured matrix scaling convex problem with ridge regularization.
    """
    n, k_plus_1 = X.shape
    k = k_plus_1 - 1
    
    primal_loss = 0.0
    dual_loss = 0.0
    dual_lin = 0.0
    s_W = np.zeros((k, k_plus_1))
    
    logits = np.zeros(k)
    exp_logits = np.zeros(k)

    for i in range(n):
        xi, yi = X[i], y[i]

        sum_exp = _compute_logits(
            xi, W_delta, k, k_plus_1, logits, exp_logits
        )

        for j in range(k):
            prob = exp_logits[j] / sum_exp
            alpha_ij = prob
            dual_loss -= prob * math.log(prob)
            if j == yi:
                alpha_ij -= 1.0
                primal_loss -= math.log(prob)
            for l in range(k_plus_1):
                s_W[j, l] += alpha_ij * xi[l]
            dual_lin += alpha_ij * xi[j]

    primal_reg = 0.0
    for j in range(k):
        primal_reg += reg_intercept * W_delta[j, k]**2
        primal_reg += reg_diagonal * W_delta[j, j]**2
        for l in range(j):
            primal_reg += reg_off_diagonal * W_delta[j, l]**2
            primal_reg += reg_off_diagonal * W_delta[l, j]**2

    primal_obj = (primal_loss / n) + primal_reg

    dual_reg = 0.0
    n_squared = float(n * n)
    factor_intercept = 1.0 / (4.0 * reg_intercept * n_squared)
    factor_diagonal = 1.0 / (4.0 * reg_diagonal * n_squared)
    factor_off_diagonal = 1.0 / (4.0 * reg_off_diagonal * n_squared)
    for j in range(k):
        dual_reg -= factor_intercept * s_W[j, k]**2
        dual_reg -= factor_diagonal * s_W[j, j]**2
        for l in range(j):
            dual_reg -= factor_off_diagonal * s_W[j, l]**2
            dual_reg -= factor_off_diagonal * s_W[l, j]**2

    dual_obj = (dual_loss / n) + (dual_lin / n) + dual_reg
    
    return primal_obj, dual_obj

@njit(fastmath=True, cache=True)
def _compute_primal_dual_group_lasso(
    X: np.ndarray, y: np.ndarray, W_delta: np.ndarray,
    reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float
) -> Tuple[float, float]:
    """Computes primal and (approximate) dual objectives for our structured matrix scaling convex problem with group-LASSO regularization.
    """
    n, k_plus_1 = X.shape
    k = k_plus_1 - 1
    
    primal_loss = 0.0
    dual_loss = 0.0
    dual_lin = 0.0
    
    logits = np.zeros(k)
    exp_logits = np.zeros(k)

    for i in range(n):
        xi, yi = X[i], y[i]

        sum_exp = _compute_logits(
            xi, W_delta, k, k_plus_1, logits, exp_logits
        )
        
        for j in range(k):
            prob = exp_logits[j] / sum_exp
            alpha_ij = prob
            dual_loss -= prob * math.log(prob)
            if j == yi:
                alpha_ij -= 1.0
                primal_loss -= math.log(prob)
            dual_lin += alpha_ij * xi[j]

    norm_intercept_sq = 0.0
    norm_diagonal_sq = 0.0
    norm_off_diagonal_sq = 0.0
    for j in range(k):
        norm_intercept_sq += W_delta[j, k] ** 2
        norm_diagonal_sq += W_delta[j, j] ** 2
        for l in range(j):
            norm_off_diagonal_sq += W_delta[j, l] ** 2 + W_delta[l, j] ** 2

    primal_reg = (reg_intercept * math.sqrt(norm_intercept_sq) + 
                  reg_diagonal * math.sqrt(norm_diagonal_sq) +
                  reg_off_diagonal * math.sqrt(norm_off_diagonal_sq))

    primal_obj = (primal_loss / n) + primal_reg

    dual_obj = (dual_loss / n) + (dual_lin / n)

    return primal_obj, dual_obj

@njit(fastmath=True, cache=True)
def _compute_primal_dual_lasso(
    X: np.ndarray, y: np.ndarray, W_delta: np.ndarray,
    reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float
) -> Tuple[float, float]:
    """Computes primal and (approximate) dual objectives for our structured matrix scaling convex problem with LASSO regularization.
    """
    n, k_plus_1 = X.shape
    k = k_plus_1 - 1
    
    primal_loss = 0.0
    dual_loss = 0.0
    dual_lin = 0.0
    
    logits = np.zeros(k)
    exp_logits = np.zeros(k)

    for i in range(n):
        xi, yi = X[i], y[i]

        sum_exp = _compute_logits(
            xi, W_delta, k, k_plus_1, logits, exp_logits
        )
        
        for j in range(k):
            prob = exp_logits[j] / sum_exp
            alpha_ij = prob
            dual_loss -= prob * math.log(prob)
            if j == yi:
                alpha_ij -= 1.0
                primal_loss -= math.log(prob)
            dual_lin += alpha_ij * xi[j]

    primal_reg = 0.0
    for j in range(k):
        primal_reg += reg_intercept * np.abs(W_delta[j, k])
        primal_reg += reg_diagonal * np.abs(W_delta[j, j])
        for l in range(j):
            primal_reg += reg_off_diagonal * np.abs(W_delta[j, l])
            primal_reg += reg_off_diagonal * np.abs(W_delta[l, j])

    primal_obj = (primal_loss / n) + primal_reg

    dual_obj = (dual_loss / n) + (dual_lin / n)

    return primal_obj, dual_obj

@njit(fastmath=True, cache=True)
def _compute_primal_dual_mcp(
    X: np.ndarray, y: np.ndarray, W_delta: np.ndarray,
    reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float
) -> Tuple[float, float]:
    """Computes primal objective for our structured matrix scaling convex problem with MCP regularization.
    With MCP penalty, the dual cannot be evaluated. Only primal is used for stopping.
    """
    n, k_plus_1 = X.shape
    k = k_plus_1 - 1
    
    primal_loss = 0.0

    logits = np.zeros(k)
    exp_logits = np.zeros(k)

    for i in range(n):
        xi, yi = X[i], y[i]

        sum_exp = _compute_logits(
            xi, W_delta, k, k_plus_1, logits, exp_logits
        )
        
        primal_loss -= math.log(exp_logits[yi] / sum_exp)

    primal_reg = 0.0
    for j in range(k):
        primal_reg += _mcp_penalty(W_delta[j, k], reg_intercept)
        primal_reg += _mcp_penalty(W_delta[j, j], reg_diagonal)
        for l in range(j):
            primal_reg += _mcp_penalty(W_delta[j, l], reg_off_diagonal)
            primal_reg += _mcp_penalty(W_delta[l, j], reg_off_diagonal)

    primal_obj = (primal_loss / n) + primal_reg

    return primal_obj, -np.inf

@njit(fastmath=True, cache=True)
def _compute_primal_dual_none(
    X: np.ndarray, y: np.ndarray, W_delta: np.ndarray,
    reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float
) -> Tuple[float, float]:
    """Computes primal objective for our structured matrix scaling convex problem without regularization.
    Since no regularization is applied, the dual cannot be evaluated. Only primal is used for stopping.
    """
    n, k_plus_1 = X.shape
    k = k_plus_1 - 1
    
    primal_loss = 0.0

    logits = np.zeros(k)
    exp_logits = np.zeros(k)

    for i in range(n):
        xi, yi = X[i], y[i]

        sum_exp = _compute_logits(
            xi, W_delta, k, k_plus_1, logits, exp_logits
        )
        
        primal_loss -= math.log(exp_logits[yi] / sum_exp)
        
    primal_obj = primal_loss / n

    return primal_obj, -np.inf

# SAGA solver
@njit(fastmath=True, cache=True)
def _saga_matrix_scaling(
    X: np.ndarray, y: np.ndarray, reg_intercept: float, reg_diagonal: float, reg_off_diagonal: float,
    max_iter: int, tol: float, prox_op: Callable[[np.ndarray, int, float, float, float, float], None],  stop_op: Callable[..., float]
) -> Tuple[float, np.ndarray, int]:
    """Solves our structured matrix scaling convex optimization problem using SAGA.

    Args:
        X (np.ndarray): un-calibrated logits.
        y (np.ndarray): un-calibrated labels (not one-hot encoded).
        reg_intercept (float): regularization strength applied to intercept parameters.
        reg_diagonal (float): regularization strength applied to diagonal parameters.
        reg_off_diagonal (float): regularization strength applied to off-diagonal parameters.
        max_iter (int): max number of dataset passes before SAGA is stopped.
        tol (float): tolerance used for stopping.
            Stopping is based on relative difference between primal and (approximate) dual objectives when available.
            Otherwise relative difference between consecutive primal objectives is used.
        prox_op (Callable[[np.ndarray, int, float, float], None]): proximal operator used to regularize.
        stop_op (Callable[..., float]): function that computes the primal and dual objectives for stopping.

    Returns:
        Tuple[float, np.ndarray, int]: global scaling parameter a, weight matrix W, number of epochs performed.
    """
    n, k_plus_1 = X.shape
    k = k_plus_1 - 1

    W_delta = np.zeros((k, k_plus_1))

    previous_objective = np.inf
    samples_seen = 1

    gradient_weights_memory = np.zeros((n, k))
    avg_gradient = np.zeros((k, k_plus_1))
    permutation = np.arange(n, dtype=np.int32)
    
    logits = np.zeros(k)
    exp_logits = np.zeros(k)

    L = L_FACTOR * np.max(np.sum(X * X, axis=1))
    step_size = 1.0 / (ETA_DIVISOR * L)

    for epoch in range(1, max_iter + 1):
        _fast_shuffle(permutation)
        
        for i in permutation:
            xi, yi = X[i], y[i]

            sum_exp = _compute_logits(
                xi, W_delta, k, k_plus_1, logits, exp_logits
            )

            for j in range(k):
                prob = exp_logits[j] / sum_exp
                gradient_weight = prob - (1.0 if j == yi else 0.0)
                
                delta_grad = gradient_weight - gradient_weights_memory[i, j]
                gradient_weights_memory[i, j] = gradient_weight
                
                for l in range(k_plus_1):
                    update = delta_grad * xi[l]
                    W_delta[j, l] -= step_size * (update + avg_gradient[j, l])
                    if epoch == 1:
                        avg_gradient[j, l] += (update - avg_gradient[j, l]) / samples_seen
                    else:
                        avg_gradient[j, l] += update / n

            if epoch == 1:
                samples_seen += 1

            prox_op(W_delta, k, step_size, reg_intercept, reg_diagonal, reg_off_diagonal)

        primal_obj, dual_obj = stop_op(X, y, W_delta, reg_intercept, reg_diagonal, reg_off_diagonal)
        if dual_obj == -np.inf:
            gap = abs(primal_obj - previous_objective)
            if gap <= tol * max(1.0, abs(primal_obj)):
                return W_delta, epoch
            previous_objective = primal_obj
        else:
            gap = abs(primal_obj - dual_obj)
            if gap <= tol * max(1.0, abs(primal_obj)):
                return W_delta, epoch

    return W_delta, max_iter

# Dictionnary mappings
MATRIX_PROX_MAP = {
    "mcp": _prox_mcp,
    "lasso": _prox_lasso,
    "group_lasso": _prox_group_lasso,
    "ridge": _prox_ridge,
    None: _prox_none,
}

MATRIX_STOP_MAP = {
    "mcp": _compute_primal_dual_mcp,
    "lasso": _compute_primal_dual_lasso,
    "group_lasso": _compute_primal_dual_group_lasso,
    "ridge": _compute_primal_dual_ridge,
    None: _compute_primal_dual_none,
}

# Public API functions
def fit_matrix_scaling(
    X: np.ndarray, y: np.ndarray, init_scaling: float, penalty: str, reg_intercept: float,
    reg_diagonal: float, reg_off_diagonal: float, max_iter: int, tol: float
) -> np.ndarray:
    """Public API to fit matrix scaling using SAGA."""
    k = X.shape[1] - 1
    try:
        prox_op = MATRIX_PROX_MAP[penalty]
        stop_op = MATRIX_STOP_MAP[penalty]
    except KeyError:
        raise ValueError(f"Unknown penalty: {penalty}")

    W_delta, last_iter = _saga_matrix_scaling(
        X, y, reg_intercept, reg_diagonal, reg_off_diagonal,
        max_iter, tol, prox_op, stop_op
    )
    if last_iter == max_iter:
        warnings.warn("SAGA reached max_iter without convergence", RuntimeWarning)

    return W_delta + np.hstack([np.eye(k), np.zeros((k, 1))])

def warm_up_matrix_scaling():
    """Public API to warm up matrix scaling."""
    X = np.random.randn(10, 3).astype(np.float32)
    y = np.random.randint(0, 3, size=10)
    for penalty in [None, 'mcp', 'lasso', 'ridge']:
        _saga_matrix_scaling(X, y, 3.0/10.0, 3.0/10.0, 2.0*3.0/10.0, 10, 1e-5, MATRIX_PROX_MAP[penalty], MATRIX_STOP_MAP[penalty])
