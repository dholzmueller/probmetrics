import copy
import numpy as np

def join_dicts(*dicts, allow_overlap: bool = True):
    # Attention: arguments do not commute since later dicts can override entries from earlier dicts!
    result = copy.copy(dicts[0])
    for d in dicts[1:]:
        if not allow_overlap and any([key in result for key in d.keys()]):
            raise ValueError(f'Overlapping keys with allow_overlap=False')
        result.update(d)
    return result

def validate_probabilities(X: np.ndarray) -> None:
    """
    Checks if X is a valid 2D array of probability distributions.
    Raises ValueError if the validation fails.
    """
    if X.ndim != 2:
        raise ValueError(f"Input array must be 2D, but got {X.ndim} dimensions.")
    if np.any(X < 0) or np.any(X > 1):
        raise ValueError("All probability values must be within the [0, 1] range.")
    if not np.all(np.isclose(X.sum(axis=1), 1.0)):
        raise ValueError("Each row of the input array must sum to 1 to be a valid probability distribution.")

def binary_probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Converts binary probabilities to logits using the logit function (inverse sigmoid).
    Clips logits to the log of the tinyest normal float32 to avoid infinite logit values.
    """
    probs = probs.astype(np.float32)
    # 1D array of probabilities
    if probs.ndim == 1:
        logits = probs
    # 2D array of probability pairs [(p0, p1), ...]
    elif probs.ndim == 2 and probs.shape[1] == 2:
        logits = probs[:, 1]
    else:
        raise ValueError(f"Invalid input shape: {probs.shape}."
                         f"1D array or 2D array with shape (n, 2)")
    
    with np.errstate(divide="ignore"):
        logits = np.log(logits) - np.log1p(-logits)
    thresh = np.log(np.finfo(np.float32).tiny)
    return np.clip(logits, a_min=thresh, a_max=-thresh)

def multiclass_probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Converts multiclass probabilities to logits using the log function.
    Clips logits to the log of the tinyest normal float32 to avoid infinite logit values.
    """
    # probs = np.atleast_2d(probs) # Ensure we have a 2D array
    probs = probs.astype(np.float32)
    thresh = np.log(np.finfo(np.float32).tiny)
    with np.errstate(divide="ignore"):
        logits = np.log(probs)
    logits = np.clip(logits, a_min=thresh, a_max=None)
    return logits
