from typing import Tuple

import torch


def remove_missing_classes(y_pred: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Removes missing classes from y_pred and y.
    For example, if y_pred.shape[-1] == 4 but y only contains the values 0 and 2,
    the columns y_pred[..., 1] and y_pred[..., 3] will be removed and the values (0, 2) will be mapped to (0, 1).
    :param y_pred: Predictions of shape (n_samples, n_classes) (should be logits
    because probabilities will not be normalized anymore after removing columns).
    :param y: classes of shape (n_samples,)
    :return: y_pred and y with missing classes removed
    """
    # shapes: y_pred should be n_samples x n_classes, y should be n_samples
    n_classes = y_pred.shape[-1]
    counts = torch.bincount(y, minlength=n_classes)
    is_present = counts > 0
    if torch.all(is_present).item():
        # all classes are present, nothing needs to be removed
        return y_pred, y

    num_present = is_present.sum().item()
    reduced_y_pred = y_pred[..., is_present]
    class_mapping = torch.zeros(n_classes, dtype=torch.long, device=y.device)
    class_mapping[is_present] = torch.arange(num_present, dtype=torch.long, device=y.device)
    reduced_y = class_mapping[y]
    # print(f'{is_present=}, {reduced_y_pred.shape=}, {torch.unique(reduced_y)=}')
    return reduced_y_pred, reduced_y
