import warnings
from typing import Callable, Optional, List

import torch
import sklearn
import logging
import numpy as np
from probmetrics.saga import (
    fit_affine_scaling,
    warm_up_affine_scaling,
    fit_quadratic_scaling,
    warm_up_quadratic_scaling,
    fit_vector_scaling,
    warm_up_vector_scaling,
    fit_matrix_scaling,
    warm_up_matrix_scaling
)
from probmetrics.utils import binary_probs_to_logits, multiclass_probs_to_logits, validate_probabilities
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from torch import nn

from probmetrics.distributions import CategoricalDistribution, CategoricalProbs, CategoricalLogits

logger = logging.getLogger(__name__)

class Calibrator(BaseEstimator, ClassifierMixin):
    """
    Calibrator base class. To implement,
    - override at least one of _fit_impl and _fit_torch_impl
    - override at least one of predict_proba and predict_proba_torch
    """

    def __init__(self):
        assert self.__class__.fit == Calibrator.fit
        assert self.__class__.fit_torch == Calibrator.fit_torch

    def fit(self, X, y) -> 'Calibrator':
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        self.classes_ = list(range(X.shape[-1]))

        if self.__class__._fit_impl != Calibrator._fit_impl:
            self._fit_impl(X, y)
            return self

        if self.__class__._fit_torch_impl != Calibrator._fit_torch_impl:
            self._fit_torch_impl(y_pred=CategoricalProbs(torch.as_tensor(X)),
                                 y_true_labels=torch.as_tensor(y, dtype=torch.long))
            return self

        raise NotImplementedError()

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError()

    def fit_torch(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor) -> 'Calibrator':
        assert isinstance(y_true_labels, torch.Tensor)
        assert isinstance(y_pred, CategoricalDistribution)
        # default implementation, using sklearn
        self.classes_ = list(range(y_pred.get_n_classes()))

        if self.__class__._fit_torch_impl != Calibrator._fit_torch_impl:
            self._fit_torch_impl(y_pred, y_true_labels)
            return self

        if self.__class__._fit_impl != Calibrator._fit_impl:
            self._fit_impl(y_pred.get_probs().detach().cpu().numpy(), y_true_labels.detach().cpu().numpy())
            return self

        raise NotImplementedError()

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        raise NotImplementedError()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.__class__.predict_proba_torch != Calibrator.predict_proba_torch:
            return self.predict_proba_torch(CategoricalProbs(torch.as_tensor(X))).get_probs().numpy()

        raise NotImplementedError()

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        if self.__class__.predict_proba != Calibrator.predict_proba:
            y_pred_probs = y_pred.get_probs()
            probs = self.predict_proba(y_pred_probs.detach().cpu().numpy())
            return CategoricalProbs(torch.as_tensor(probs, device=y_pred_probs.device, dtype=y_pred_probs.dtype))

        raise NotImplementedError()

    def predict(self, X):
        y_probs = self.predict_proba(X)
        class_idxs = np.argmax(y_probs, axis=-1)
        return np.asarray(self.classes_)[class_idxs]


class ApplyToLogitsCalibrator(Calibrator):
    def __init__(self, calib: BaseEstimator):
        super().__init__()
        self.calib = calib

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self.calib_ = sklearn.base.clone(self.calib)
        self.calib_.fit(np.log(X + 1e-10), y)

    def predict_proba(self, X):
        return self.calib_.predict_proba(np.log(X + 1e-10))


def bisection_search(f: Callable[[float], float], a: float, b: float, n_steps: int):
    for _ in range(n_steps):
        c = a + 0.5 * (b - a)
        f_c = f(c)
        if f_c > 0:
            b = c
        else:
            a = c

    return 0.5 * (a + b)


class TemperatureScalingCalibrator(Calibrator):
    def __init__(self, opt: str = 'bisection', max_bisection_steps: int = 30, lr: float = 0.1, max_iter: int = 200,
                 use_inv_temp: bool = True, inv_temp_init: float = 1 / 1.5):
        super().__init__()
        self.lr = lr
        self.max_bisection_steps = max_bisection_steps
        self.max_iter = max_iter
        self.use_inv_temp = use_inv_temp
        self.inv_temp_init = inv_temp_init
        self.opt = opt

    def _get_loss_grad(self, invtemp: float, logits: torch.Tensor, y: torch.Tensor):
        part_1 = torch.mean(torch.sum(logits * torch.softmax(invtemp * logits, dim=-1), dim=-1))
        part_2 = torch.mean(logits[torch.arange(logits.shape[0]), y])
        return (part_1 - part_2).item()

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        logits = y_pred.get_logits()
        labels = y_true_labels

        if self.opt in ['lbfgs', 'lbfgs_line_search']:
            self._fit_lbfgs(logits, labels)
        elif self.opt == 'bisection':
            self._fit_bisection(logits, labels)
        else:
            raise ValueError(f'Unknown optimizer "{self.opt}"')

        # print(f'{self.invtemp_=}')

    def _fit_lbfgs(self, logits: torch.Tensor, labels: torch.Tensor):
        # following https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
        param = nn.Parameter(
            torch.ones(1, device=logits.device) * (self.inv_temp_init if self.use_inv_temp else 1 / self.inv_temp_init))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([param], lr=self.lr, max_iter=self.max_iter,
                                      line_search_fn='strong_wolfe' if self.opt == 'lbfgs_line_search' else None)

        def eval():
            optimizer.zero_grad()
            y_pred = logits * param[:, None] if self.use_inv_temp else logits / param[:, None]
            loss = criterion(y_pred, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        self.invtemp_ = param.item() if self.use_inv_temp else 1 / param.item()

    def _fit_bisection(self, logits: torch.Tensor, labels: torch.Tensor):
        objective_grad = lambda u, l=logits, tar=labels: self._get_loss_grad(np.exp(u), l, tar)

        # should reach about float32 accuracy
        # need log_2(32) = 5 steps to get to length 1 and then 24 more steps to get to float32 epsilon (2^{-24})
        self.invtemp_ = np.exp(bisection_search(objective_grad, a=-16, b=16, n_steps=self.max_bisection_steps))
        # print(f'{self.invtemp_=}')

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        return CategoricalLogits(self.invtemp_ * y_pred.get_logits())


### New binary calibrators ###

class BinaryLogisticCalibrator(Calibrator):
    """
    Binary post-hoc calibration with linear, affine or quadratic scaling 
    on the binary logits using sklearn's LogisticRegression.

    This class fits a model of the form:
    P(y=1) = sigma(calibrated_logit)
    
    Where `calibrated_logit` depends on the `type`:
    - 'linear':    calibrated_logit = w * logit (Temperature scaling with inverse temperature)
    - 'affine':    calibrated_logit = b + w * logit (Platt scaling)
    - 'quadratic': calibrated_logit = b + w1 * logit + w2 * logit^2
    """

    VALID_TYPES = ['linear', 'affine', 'quadratic']

    def __init__(self, type: str = 'affine'):
        """
        Args:
            type (str, optional): The type of scaling. 
                One of ['linear', 'affine', 'quadratic']. 
                Defaults to 'affine'.
        """
        super().__init__()
        
        if type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid type '{type}'. Must be one of {self.VALID_TYPES}"
            )
        self.type = type
        self.cal_ = None

    def _get_logits(self, X: np.ndarray) -> np.ndarray:
        logits = binary_probs_to_logits(X)
        logits = logits.reshape(-1,1)
        if self.type == 'affine':
            # Adding a constant term to logit vectors.
            logits = np.hstack([np.ones((len(X), 1)), logits])
        elif self.type == 'quadratic':
            # Adding a constant and quadratic term to logit vectors.
            logits = np.hstack([np.ones((len(X), 1)), logits, np.square(logits)])
        return logits

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the logistic regression calibrator."""
        from sklearn.linear_model import LogisticRegression
        self.cal_ = LogisticRegression(penalty=None, fit_intercept=False)

        features = self._get_logits(X)
        self.cal_.fit(features, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generates calibrated probability estimates."""
        if self.cal_ is None:
            raise RuntimeError("Calibrator has not been fitted yet. Call fit() before predict_proba().")

        features = self._get_logits(X)
        return self.cal_.predict_proba(features)

class AffineScalingCalibrator(Calibrator):
    """
    Binary post-hoc calibration with Platt scaling (~binary logistic regression) $sigmoid(ax+b)$ on the logits x.
    Uses the SAGA algorithm to fit the scaling and intercept parameters a & b.
    Numba functions are warmed-up the first time the class is initialized (~1s).
    A penalty (one of ["mcp", "lasso", "ridge"]) can be applied to the intercept parameter b, with regularization strength lambda_intercept.
    To apply no regularization, leave penalty to None.
    """
    _warmed_up = False

    def __init__(
        self,
        penalty: str = None,
        lambda_intercept: float = 0.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        print_init_info: bool = True
    ):
        super().__init__()
        self.penalty = penalty
        self.lambda_intercept = lambda_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.print_init_info = print_init_info

        self.w = None

        if not AffineScalingCalibrator._warmed_up:
            if self.print_init_info:
                print("First AffineScalingCalibrator instantiation - warming up Numba functions...")
            warm_up_affine_scaling()
            AffineScalingCalibrator._warmed_up = True
            if self.print_init_info:
                print("Warmed up!")

    def _get_logits(self, X: np.ndarray) -> np.ndarray: 
        logits = binary_probs_to_logits(X)
        logits = logits.reshape(-1,1)
        # Adding a constant term to logit vectors to fit the intercept b.
        logits = np.hstack([np.ones((len(X), 1)), logits])
        return logits

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        logits = self._get_logits(X)
        self.w = fit_affine_scaling(
            logits, y,
            penalty=self.penalty,
            reg_intercept=self.lambda_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Calibrator has not been fitted yet. Call fit() before predict_proba().")

        logits = self._get_logits(X)
        logits = logits.dot(self.w).reshape(-1,1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.hstack([1-probs, probs])


class QuadraticScalingCalibrator(Calibrator):
    """
    Binary post-hoc calibration with quadratic scaling $sigmoid(a + bx + cx^2)$ on the logits x.
    Uses the SAGA algorithm to fit the intercept, scaling and quadratic parameters a, b & c.
    Numba functions are warmed-up the first time the class is initialized (~1s).
    A penalty (one of ["mcp", "lasso", "ridge"]) can be applied to the intercept and quadratic parameters a & c,
    with respective regularization strength lambda_intercept and lambda_quadratic.
    To apply no regularization, leave penalty to None.
    """
    _warmed_up = False

    def __init__(
        self,
        penalty: str = None,
        lambda_intercept: float = 0.0,
        lambda_quadratic: float = 0.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        print_init_info: bool = True
    ):
        super().__init__()
        self.penalty = penalty
        self.lambda_intercept = lambda_intercept
        self.lambda_quadratic = lambda_quadratic
        self.max_iter = max_iter
        self.tol = tol
        self.print_init_info = print_init_info

        self.w = None

        if not QuadraticScalingCalibrator._warmed_up:
            if self.print_init_info:
                print("First QuadraticScalingCalibrator instantiation - warming up Numba functions...")
            warm_up_quadratic_scaling()
            QuadraticScalingCalibrator._warmed_up = True
            if self.print_init_info:
                print("Warmed up!")

    def _get_logits(self, X: np.ndarray) -> np.ndarray: 
        logits = binary_probs_to_logits(X)
        logits = logits.reshape(-1,1)
        # Adding a constant and quadratic term to logit vectors.
        logits = np.hstack([np.ones((len(X), 1)), logits, np.square(logits)])
        return logits

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:        
        logits = self._get_logits(X)
        self.w = fit_quadratic_scaling(
            logits, y,
            penalty=self.penalty,
            reg_intercept=self.lambda_intercept,
            reg_quadratic=self.lambda_quadratic,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Calibrator has not been fitted yet. Call fit() before predict_proba().")

        logits = self._get_logits(X)
        logits = logits.dot(self.w).reshape(-1,1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.hstack([1-probs, probs])

##################################


### New multiclass calibrators ###

class SVSCalibrator(Calibrator):
    """
    Multiclass post-hoc calibration with a structured scaling scheme $softmax((aI + diag(v))x + b)$ on the logits x.

    Numba functions are warmed-up the first time the class is initialized with the 'saga' optimizer.
        
    A penalty (one of [None, "mcp", "lasso", "ridge"]) is applied to the intercept (b) and vector scaling (v) parameters,
    with respective regularization strength lambda_intercept*(k**rho)/(n**tau) and lambda_diagonal*(k**rho)/(n**tau).
    Note that the general scaling parameter (a) is not regularized, just like in classical temperature scaling.

    [!] solver = 'bfgs' only supports 'ridge' penalty.
    [!] For solver = 'bfgs', max_iter and tol are ignored and default torchmin parameters are used.
    """
    _warmed_up_saga = False
    _ALLOWED_PENALTIES = [None, "mcp", "lasso", "ridge"]
    _ALLOWED_OPTIMIZERS = ["saga", "bfgs"]

    def __init__(
        self,
        penalty: str = 'ridge',
        rho: float = 1.0,
        tau: float = 1.0,
        lambda_intercept: float = 1.0,
        lambda_diagonal: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-5,
        opt: str = 'saga',
        ts_preprocessing: bool = True,
        print_init_info: bool = True
    ):
        super().__init__()

        if opt not in self._ALLOWED_OPTIMIZERS:
            raise ValueError(f"Optimizer '{opt}' not recognized. Choose from {self._ALLOWED_OPTIMIZERS}.")
        if penalty not in self._ALLOWED_PENALTIES:
            raise ValueError(f"Penalty '{penalty}' not recognized. Choose from {self._ALLOWED_PENALTIES}.")

        if opt == 'saga' and not SVSCalibrator._warmed_up_saga:
            if print_init_info:
                logger.info("First SVSCalibrator instantiation with 'saga' - warming up Numba functions...")
            warm_up_vector_scaling()
            SVSCalibrator._warmed_up_saga = True
            if print_init_info:
                logger.info("Warmed up!")

        if opt == 'bfgs' and penalty != 'ridge':
            raise ValueError("The 'bfgs' optimizer only supports the 'ridge' penalty.")

        if opt == 'bfgs':
            try:
                import torchmin
            except ImportError:
                raise ImportError("The 'torchmin' package is required for the 'bfgs' optimizer. Please install it.")

        self.penalty = penalty
        self.rho = rho
        self.tau = tau
        self.lambda_intercept = lambda_intercept
        self.lambda_diagonal = lambda_diagonal
        self.max_iter = max_iter
        self.tol = tol
        self.opt = opt
        self.ts_preprocessing = ts_preprocessing
        self.print_init_info = print_init_info

        self.a = None
        self.v = None
        self.b = None
        self.ts = None

    def _get_logits(self, X: np.ndarray) -> tuple[np.ndarray, int, int]:
        n, k = X.shape
        logits = multiclass_probs_to_logits(X)
        return logits, n, k

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        validate_probabilities(X)

        if self.ts_preprocessing:
            self.ts = get_calibrator('ts-mix')
            self.ts.fit(X, y)
            probs_to_calibrate = self.ts.predict_proba(X)
        else:
            self.ts = None
            probs_to_calibrate = X
        
        logits, n, k = self._get_logits(probs_to_calibrate)

        self.a = 1.0
        self.v = np.zeros(k)
        self.b = np.zeros(k)

        reg_intercept = self.lambda_intercept * (k**self.rho) / (n**self.tau)
        reg_diagonal = self.lambda_diagonal * (k**self.rho) / (n**self.tau)

        if self.opt == 'saga':
            self.a, self.v, self.b = fit_vector_scaling(
                logits, y,
                init_scaling=1.0,
                penalty=self.penalty,
                reg_intercept=reg_intercept,
                reg_diagonal=reg_diagonal,
                max_iter=self.max_iter,
                tol=self.tol,
            )

        elif self.opt == 'bfgs':
            self._fit_bfgs(
                torch.tensor(logits), torch.tensor(y),
                num_classes=k,
                reg_intercept=reg_intercept,
                reg_diagonal=reg_diagonal
            )

    def _fit_bfgs(self, logits: torch.Tensor, y: torch.Tensor, num_classes, reg_intercept, reg_diagonal):
        import torchmin

        a = torch.tensor([1.0], device=logits.device, requires_grad=True)
        v = torch.zeros(num_classes, device=logits.device, requires_grad=True)
        b = torch.zeros(num_classes, device=logits.device, requires_grad=True)

        params = torch.cat([a, v, b])

        def loss(params):
            a, v, b = params.split([1, num_classes, num_classes])
            logloss = torch.nn.functional.cross_entropy((a + v) * logits + b, y)
            reg = reg_intercept * torch.dot(b, b) + reg_diagonal * torch.dot(v, v)
            return logloss + reg

        res = torchmin.bfgs._minimize_bfgs(loss, params)

        a, v, b = res.x.split([1, num_classes, num_classes])

        self.a = a.item()
        self.v = v.detach().cpu().numpy()
        self.b = b.detach().cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        validate_probabilities(X)

        if self.a is None:
            raise RuntimeError("Calibrator has not been fitted yet. Call fit() before predict_proba().")

        if self.ts_preprocessing:
            probs_to_calibrate = self.ts.predict_proba(X) 
        else:
            probs_to_calibrate = X
        
        logits, _, _ = self._get_logits(probs_to_calibrate)
        
        calibrated_logits = (self.a + self.v.reshape(1,-1)) * logits + self.b.reshape(1,-1)
        calibrated_logits -= np.max(calibrated_logits, axis=1, keepdims=True)
        exp_logits = np.exp(calibrated_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class SMSCalibrator(Calibrator):
    """
    Multiclass post-hoc calibration with a structured scaling scheme $softmax( (aI + diag(v) + off_diag(M) )x + b)$ on the logits x.

    Numba functions are warmed-up the first time the class is initialized with the 'saga' optimizer.

    A penalty (one of ["mcp", "lasso", "group_lasso", "ridge"]) is applied to the intercept (b), vector scaling (v) and off diagonal (M) parameters,
    with respective regularization strength lambda_intercept*(k**rho)/(n**tau), lambda_diagonal*(k**rho)/(n**tau) and lambda_off_diagonal*((k*(k-1))**rho)/(n**tau).
    Note that the general scaling parameter (a) is not regularized, just like in classical temperature scaling.

    [!] The 'bfgs' solver only supports 'ridge' penalty.
    [!] If 'bfgs' solver is used, max_iter and tol are ignored and default torchmin parameters are used.
    """
    _warmed_up_saga = False
    _ALLOWED_PENALTIES = [None, "mcp", "lasso", "group_lasso", "ridge"]
    _ALLOWED_OPTIMIZERS = ["saga", "bfgs"]

    def __init__(
        self,
        penalty: str = 'ridge',
        rho: float = 1.0,
        tau: float = 1.0,
        lambda_intercept: float = 1.0,
        lambda_diagonal: float = 1.0,
        lambda_off_diagonal: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-5,
        opt: str = 'saga',
        ts_preprocessing: bool = True,
        print_init_info: bool = True
    ):
        super().__init__()

        if opt not in self._ALLOWED_OPTIMIZERS:
            raise ValueError(f"Optimizer '{opt}' not recognized. Choose from {self._ALLOWED_OPTIMIZERS}.")
        if penalty not in self._ALLOWED_PENALTIES:
            raise ValueError(f"Penalty '{penalty}' not recognized. Choose from {self._ALLOWED_PENALTIES}.")

        if opt == 'saga' and not SMSCalibrator._warmed_up_saga:
            if print_init_info:
                logger.info("First SMSCalibrator instantiation with 'saga' - warming up Numba functions...")
            warm_up_matrix_scaling()
            SMSCalibrator._warmed_up_saga = True
            if print_init_info:
                logger.info("Warmed up!")

        if opt == 'bfgs' and penalty != 'ridge':
            raise ValueError("The 'bfgs' optimizer only supports the 'ridge' penalty.")

        if opt == 'bfgs':
            try:
                import torchmin
            except ImportError:
                raise ImportError("The 'torchmin' package is required for the 'bfgs' optimizer. Please install it.")

        self.penalty = penalty
        self.rho = rho
        self.tau = tau
        self.lambda_intercept = lambda_intercept
        self.lambda_diagonal = lambda_diagonal
        self.lambda_off_diagonal = lambda_off_diagonal
        self.max_iter = max_iter
        self.tol = tol
        self.opt = opt
        self.ts_preprocessing = ts_preprocessing
        self.print_init_info = print_init_info

        self.W = None
        self.ts = None

    def _get_logits(self, X: np.ndarray) -> tuple[np.ndarray, int, int]:
        n, k = X.shape
        logits = multiclass_probs_to_logits(X)
        # Adding a constant term to logit vectors to fit the intercept.
        logits = np.hstack([logits, np.ones((n, 1))])
        return logits, n, k

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the calibration matrix W."""
        validate_probabilities(X)

        if self.ts_preprocessing:
            self.ts = get_calibrator('ts-mix')
            self.ts.fit(X, y)
            probs_to_calibrate = self.ts.predict_proba(X)
        else:
            self.ts = None
            probs_to_calibrate = X

        logits, n, k = self._get_logits(probs_to_calibrate)

        self.W = np.hstack([np.eye(k), np.zeros((k, 1))])

        reg_intercept=self.lambda_intercept * (k**self.rho) / (n**self.tau)
        reg_diagonal=self.lambda_diagonal * (k**self.rho) / (n**self.tau)
        reg_off_diagonal=self.lambda_off_diagonal * ((k*(k-1))**self.rho) / (n**self.tau)

        if self.opt == 'saga':
            self.W = fit_matrix_scaling(
                logits, y,
                penalty=self.penalty, max_iter=self.max_iter, tol=self.tol, init_scaling=1.0,
                reg_intercept=reg_intercept, reg_diagonal=reg_diagonal, reg_off_diagonal=reg_off_diagonal
            )

        elif self.opt == 'bfgs':
            self._fit_bfgs(
                torch.tensor(logits), torch.tensor(y), num_classes=k,
                reg_intercept=reg_intercept, reg_diagonal=reg_diagonal, reg_off_diagonal=reg_off_diagonal
            )

    def _fit_bfgs(self, logits: torch.Tensor, y: torch.Tensor, num_classes, reg_intercept, reg_diagonal, reg_off_diagonal):
        import torchmin

        a = torch.tensor([1.0], device=logits.device, dtype=logits.dtype)
        W = torch.zeros((num_classes, num_classes + 1), device=logits.device, dtype=logits.dtype)

        diag_mask = torch.hstack([torch.eye(num_classes), torch.zeros((num_classes, 1))])
        off_diag_mask = torch.hstack([torch.ones((num_classes, num_classes)) - torch.eye(num_classes), torch.zeros((num_classes, 1))])
        intercept_mask = torch.hstack([torch.zeros((num_classes, num_classes)), torch.ones((num_classes, 1))])

        def loss(params):
            a, W = params.split([1, num_classes*(num_classes+1)])
            a = a.squeeze()
            W = W.view(num_classes, num_classes+1)
            loss = torch.nn.functional.cross_entropy(logits @ (a*diag_mask + W).T, y)
            reg = reg_intercept*torch.square(intercept_mask * W).sum()
            reg += reg_diagonal*torch.square(diag_mask * W).sum()
            reg += reg_off_diagonal*torch.square(off_diag_mask * W).sum()
            return loss + reg

        params = torch.cat([a, W.view(-1)])
        method = "l-bfgs" if params.numel() > 1000 else "bfgs"
        res = torchmin.minimize(loss, params, method=method)

        a, W = res.x.split([1, num_classes*(num_classes+1)])
        W = W.view(num_classes, num_classes+1)

        self.W = a.item()*diag_mask.numpy() + W.detach().cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        validate_probabilities(X)

        if self.W is None:
            raise RuntimeError("Calibrator has not been fitted yet. Call fit() before predict_proba().")

        if self.ts_preprocessing:
            probs_to_calibrate = self.ts.predict_proba(X) 
        else:
            probs_to_calibrate = X
        
        logits, _, _ = self._get_logits(probs_to_calibrate)
        calibrated_logits = logits @ self.W.T
        calibrated_logits -= np.max(calibrated_logits, axis=1, keepdims=True)
        exp_logits = np.exp(calibrated_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

##################################


### Default logistic calibrator, compatible with binary and multiclass ###

class LogisticCalibrator(Calibrator):
    """
    By default, uses 'affine-scaling' for binary classification and 'sms' for multiclass classification.
    """

    VALID_BINARY_TYPES = ['linear', 'affine', 'quadratic']
    VALID_MULTICLASS_TYPES = ['svs', 'sms']

    def __init__(self, binary_type: str = 'affine', multiclass_type: str = 'sms'):
        super().__init__()
        
        if binary_type not in self.VALID_BINARY_TYPES:
            raise ValueError(
                f"Invalid binary type '{binary_type}'. Must be one of {self.VALID_BINARY_TYPES}"
            )
        self.binary_type = binary_type

        if multiclass_type not in self.VALID_MULTICLASS_TYPES:
            raise ValueError(
                f"Invalid multiclass type '{multiclass_type}'. Must be one of {self.VALID_MULTICLASS_TYPES}"
            )
        self.multiclass_type = multiclass_type

        self.cal_ = None

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        n_classes = len(np.unique(y))
        
        if n_classes == 2:
            if self.binary_type == 'affine':
                self.cal_ = BinaryLogisticCalibrator(type='affine')
            elif self.binary_type == 'linear':
                self.cal_ = BinaryLogisticCalibrator(type='linear')
            elif self.binary_type == 'quadratic':
                self.cal_ = BinaryLogisticCalibrator(type='quadratic')

        else:
            if self.multiclass_type == 'sms':
                self.cal_ = SMSCalibrator(opt='bfgs')
            elif self.multiclass_type == 'svs':
                self.cal_ = SVSCalibrator(opt='bfgs')

        self.cal_.fit(X, y)
        return self

    def predict(self, X):
        if self.cal_ is None:
            raise RuntimeError("LogisticCalibrator must be fitted before prediction.")
        return self.cal_.predict(X)

    def predict_proba(self, X):
        if self.cal_ is None:
            raise RuntimeError("LogisticCalibrator must be fitted before prediction.")
        return self.cal_.predict_proba(X)

##################################


class GuoTemperatureScalingCalibrator(Calibrator):
    # adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        logits = y_pred.get_logits()
        labels = y_true_labels

        nll_criterion = nn.CrossEntropyLoss().cuda()

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        return self

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        with torch.no_grad():
            return CategoricalLogits(y_pred.get_logits() / self.temperature)


class AutoGluonTemperatureScalingCalibrator(Calibrator):
    # adapted from
    # https://github.com/autogluon/autogluon/blob/c1181326cf6b7e3b27a7420273f1a82808d939e2/core/src/autogluon/core/calibrate/temperature_scaling.py#L9
    # https://github.com/autogluon/autogluon/blob/28a242ebe8d55ba770c991b9db153ab4623c9abd/tabular/src/autogluon/tabular/trainer/abstract_trainer.py#L4433-L4457
    def __init__(self, init_val: float = 1, max_iter: int = 200, lr: float = 0.1):
        super().__init__()
        self.init_val = init_val
        self.max_iter = max_iter
        self.lr = lr
        self.temperature = init_val

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        y_val_tensor = y_true_labels
        temperature_param = torch.nn.Parameter(torch.ones(1).fill_(self.init_val))
        logits = y_pred.get_logits()

        is_invalid = torch.isinf(logits).any().tolist()
        if is_invalid:
            return

        nll_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([temperature_param], lr=self.lr, max_iter=self.max_iter)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        optimizer_trajectory = []

        if self.init_val != 1.0:
            # need to check 1.0 as well since AutoGluon does it outside
            optimizer_trajectory.append((nll_criterion(logits, y_val_tensor).item(), 1.0))

        def temperature_scale_step():
            optimizer.zero_grad()
            temp = temperature_param.unsqueeze(1).expand(logits.size(0), logits.size(1))
            new_logits = logits / temp
            loss = nll_criterion(new_logits, y_val_tensor)
            loss.backward()
            scheduler.step()
            optimizer_trajectory.append((loss.item(), temperature_param.item()))
            return loss

        optimizer.step(temperature_scale_step)

        try:
            best_loss_index = np.nanargmin(np.array(optimizer_trajectory)[:, 0])
        except ValueError:
            self.temperature = 1.0
            return
        temperature_scale = float(np.array(optimizer_trajectory)[best_loss_index, 1])

        if np.isnan(temperature_scale) or temperature_scale <= 0.0:
            self.temperature = 1.0
            return

        self.temperature = temperature_scale

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        with torch.no_grad():
            return CategoricalLogits(y_pred.get_logits() / self.temperature)


class AutoGluonInverseTemperatureScalingCalibrator(Calibrator):
    # adapted from
    # https://github.com/autogluon/autogluon/blob/c1181326cf6b7e3b27a7420273f1a82808d939e2/core/src/autogluon/core/calibrate/temperature_scaling.py#L9
    # but optimizing the inverse temperature instead
    def __init__(self, init_val: float = 1, max_iter: int = 200, lr: float = 0.1):
        super().__init__()
        self.init_val = init_val
        self.max_iter = max_iter
        self.lr = lr
        self.inv_temp = init_val

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        y_val_tensor = y_true_labels
        inv_temperature_param = torch.nn.Parameter(torch.ones(1).fill_(self.init_val))
        logits = y_pred.get_logits()

        is_invalid = torch.isinf(logits).any().tolist()
        if is_invalid:
            return

        nll_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([inv_temperature_param], lr=self.lr, max_iter=self.max_iter)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        optimizer_trajectory = []

        def temperature_scale_step():
            optimizer.zero_grad()
            inv_temp = inv_temperature_param.unsqueeze(1).expand(logits.size(0), logits.size(1))
            new_logits = logits * inv_temp
            loss = nll_criterion(new_logits, y_val_tensor)
            loss.backward()
            scheduler.step()
            optimizer_trajectory.append((loss.item(), inv_temperature_param.item()))
            return loss

        optimizer.step(temperature_scale_step)

        try:
            best_loss_index = np.nanargmin(np.array(optimizer_trajectory)[:, 0])
        except ValueError:
            return
        inv_temperature_scale = float(np.array(optimizer_trajectory)[best_loss_index, 1])

        if np.isnan(inv_temperature_scale):
            return

        self.inv_temp = inv_temperature_scale

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        with torch.no_grad():
            return CategoricalLogits(y_pred.get_logits() * self.inv_temp)


class TorchUncertaintyTemperatureScalingCalibrator(TemperatureScalingCalibrator):
    # adapted from
    # https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/3a021d2e34e183b8aad3a0345e6d750c08c72af3/torch_uncertainty/post_processing/calibration/scaler.py#L1
    # https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/3a021d2e34e183b8aad3a0345e6d750c08c72af3/torch_uncertainty/post_processing/calibration/temperature_scaler.py#L9
    def __init__(self, init_val: float = 1, lr: float = 0.1, max_iter: int = 100):
        super().__init__(opt='lbfgs', lr=lr, max_iter=max_iter, use_inv_temp=False, inv_temp_init=1. / init_val)
        # need to save these values here to comply with sklearn cloneability conventions
        self.init_val = init_val
        self.lr = lr
        self.max_iter = max_iter


class NetcalTemperatureScalingCalibrator(Calibrator):
    # this one does nothing due to https://github.com/EFS-OpenSource/calibration-framework/issues/61
    def __init__(self):
        super().__init__()

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from netcal.scaling import TemperatureScaling

        self.cal_ = TemperatureScaling()
        if X.shape[1] == 2:
            # binary, convert
            X = X[:, 1]
        self.cal_.fit(X, y, random_state=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == 2:
            # binary, convert
            X = X[:, 1]
        result = self.cal_.transform(X)
        if len(result.shape) == 1:
            return np.stack([1. - result, result], axis=1)
        elif result.shape[1] == 1:
            return np.concatenate([1. - result, result], axis=1)

        return result


class TorchcalTemperatureScalingCalibrator(Calibrator):
    # adapted from
    # https://github.com/rishabh-ranjan/torchcal/blob/3fb65f6423d33d680cd68c7f40a0259d41e8fb0b/torchcal.py#L8
    def __init__(self):
        super().__init__()
        self.temperature = 1.0

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        import torchmin
        y_pred_logits = y_pred.get_logits()

        temp = torch.ones(1, device=y_pred_logits.device)

        def loss(t):
            return torch.nn.functional.cross_entropy(y_pred_logits / t, y_true_labels)

        res = torchmin.minimize(loss, temp, method='newton-exact')
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
        else:
            temp = res.x

        self.temperature = temp.item()

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        return CategoricalLogits(y_pred.get_logits() / self.temperature)


class TorchcalVectorScalingCalibrator(Calibrator):
    # adapted from
    # https://github.com/rishabh-ranjan/torchcal/blob/3fb65f6423d33d680cd68c7f40a0259d41e8fb0b/torchcal.py#L8
    def __init__(self):
        super().__init__()

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        import torchmin
        y_pred_logits = y_pred.get_logits()

        num_classes = y_pred_logits.shape[1]

        self.temp = torch.ones(num_classes, device=y_pred_logits.device)
        self.bias = torch.zeros(num_classes, device=y_pred_logits.device)

        def loss(temp_bias):
            temp, bias = temp_bias.split([num_classes, num_classes])
            return torch.nn.functional.cross_entropy(y_pred_logits / temp + bias, y_true_labels)
        temp_bias = torch.cat([self.temp, self.bias])
        res = torchmin.minimize(loss, temp_bias, method="bfgs")
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
        else:
            self.temp, self.bias = res.x.split([num_classes, num_classes])

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        return CategoricalLogits(y_pred.get_logits() / self.temp + self.bias)


class TorchcalMatrixScalingCalibrator(Calibrator):
    def __init__(self):
        super().__init__()

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        import torchmin
        y_pred_logits = y_pred.get_logits()

        num_classes = y_pred_logits.shape[1]

        num_params = num_classes*(num_classes + 1)
        if num_params <= 1000:
            method = "bfgs"
        else:
            method = "l-bfgs"

        self.itemp = torch.eye(num_classes, num_classes, device=y_pred_logits.device, dtype=y_pred_logits.dtype)
        self.bias = torch.zeros(num_classes, device=y_pred_logits.device, dtype=y_pred_logits.dtype)

        def loss(itemp_bias):
            itemp, bias = itemp_bias.split([num_classes**2, num_classes])
            itemp = itemp.view(num_classes, num_classes)
            return torch.nn.functional.cross_entropy(y_pred_logits @ itemp + bias, y_true_labels)

        itemp_bias = torch.cat([self.itemp.view(-1), self.bias])
        res = torchmin.minimize(loss, itemp_bias, method=method)
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
        else:
            self.itemp, self.bias = res.x.split(
                [num_classes**2, num_classes]
            )
            self.itemp = self.itemp.view(num_classes, num_classes)

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        return CategoricalLogits(y_pred.get_logits() @ self.itemp + self.bias)


class CenteredIsotonicRegressionCalibrator(Calibrator):
    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        assert not np.any(np.isnan(X))
        # print(f'{np.unique(y)=}')
        # print(f'{np.unique(X)=}')
        # we have to use float64 since with float32 it can happen that somewhere internally
        # a rounding error occurs (?) and interp1d thinks that it is asked to extrapolate at the boundary value,
        # which causes it to output nan values,
        # which are seen as a separate possible value but (y==value) is empty because nan==nan is false,
        # so an error occurs because the method attempts to take an average of an empty set
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        from cir_model import CenteredIsotonicRegression
        self.cal_ = CenteredIsotonicRegression()
        self.cal_.fit(X[:, 1], y)
        self.min_ = np.min(X[:, 1])
        self.max_ = np.max(X[:, 1])

    def predict_proba(self, X):
        # have to clip since CenteredIsotonicRegression refuses to extrapolate (?)
        X = X.astype(np.float64)
        X = np.clip(X, self.min_, self.max_)
        pred_probs = self.cal_.transform(X[:, 1])
        pred_probs = np.clip(pred_probs, 0.0, 1.0)
        return np.stack([1. - pred_probs, pred_probs], axis=-1)


class BinaryVennAbersCalibrator(Calibrator):
    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from venn_abers import VennAbers
        self.va_ = VennAbers()
        self.va_.fit(X, y)

    def predict_proba(self, X):
        return self.va_.predict_proba(X)[0]


class VennAbersCalibrator(Calibrator):
    def __init__(self, use_ovo: bool = False):
        super().__init__()
        self.use_ovo = use_ovo

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_ = np.copy(X)
        self.y_ = np.copy(y)

    def predict_proba(self, X):
        from venn_abers import VennAbersCalibrator
        va = VennAbersCalibrator()
        return va.predict_proba(p_cal=self.X_, y_cal=self.y_, p_test=X, p0_p1_output=False,
                                va_type='one_vs_one' if self.use_ovo else 'one_vs_all')


class MulticlassOneVsOneCalibrator(Calibrator):
    def __init__(self, binary_calibrator: BaseEstimator):
        super().__init__()
        self.binary_calibrator = binary_calibrator

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_classes_ = X.shape[-1]
        self.bin_calibs_ = []

        if self.n_classes_ == 2:
            # binary classification
            bin_calib = sklearn.base.clone(self.binary_calibrator)
            bin_calib.fit(X, y)
            self.bin_calibs_.append(bin_calib)
        else:
            for i in range(self.n_classes_):
                for j in range(i + 1, self.n_classes_):
                    idxs = np.logical_or((y == i), (y == j))
                    # idxs = np.arange(X.shape[0])
                    bin_probs = np.stack([X[idxs, j], X[idxs, i]], axis=-1)
                    bin_probs += 1e-30
                    bin_probs /= np.sum(bin_probs, axis=-1, keepdims=True)
                    # print(f'{np.any(np.isnan(bin_probs))=}')
                    bin_labels = (y[idxs] == i).astype(np.int32)
                    bin_calib = sklearn.base.clone(self.binary_calibrator)
                    bin_calib.fit(bin_probs, bin_labels)
                    self.bin_calibs_.append(bin_calib)

    def predict_proba(self, X):
        if self.n_classes_ == 2:
            # binary classification
            return self.bin_calibs_[0].predict_proba(X)
        else:
            # use PKPD formula in http://proceedings.mlr.press/v60/manokhin17a/manokhin17a.pdf
            pair_probs = [[None] * self.n_classes_ for i in range(self.n_classes_)]
            multi_probs = []
            calib_idx = 0
            for i in range(self.n_classes_):
                for j in range(i + 1, self.n_classes_):
                    bin_probs = np.stack([X[:, j], X[:, i]], axis=-1)
                    bin_probs += 1e-30
                    bin_probs /= np.sum(bin_probs, axis=-1, keepdims=True)
                    pred_probs = self.bin_calibs_[calib_idx].predict_proba(bin_probs)
                    pair_probs[i][j] = pred_probs[:, 1]
                    pair_probs[j][i] = pred_probs[:, 0]
                    # if i == 0 and j == 1:
                    #     print(f'{bin_probs=}')
                    #     print(f'{pred_probs=}')
                    calib_idx += 1

            for i in range(self.n_classes_):
                sum_inv_probs = sum([1. / (1e-30 + pair_probs[i][j]) for j in range(self.n_classes_) if j != i])
                multi_probs.append(1. / np.clip(sum_inv_probs - (self.n_classes_ - 2), 1e-30, np.inf))

            multi_probs = np.stack(multi_probs, axis=-1)
            multi_probs = np.clip(multi_probs, a_min=1e-30, a_max=np.inf)
            multi_probs = multi_probs / np.sum(multi_probs, axis=-1, keepdims=True)
            return multi_probs


class MulticlassOneVsRestCalibrator(Calibrator):
    def __init__(self, binary_calibrator: BaseEstimator):
        super().__init__()
        self.binary_calibrator = binary_calibrator

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_classes_ = X.shape[-1]
        self.bin_calibs_ = []

        X[np.isnan(X)] = 0.0

        if self.n_classes_ == 2:
            # binary classification
            bin_calib = sklearn.base.clone(self.binary_calibrator)
            bin_calib.fit(X, y)
            self.bin_calibs_.append(bin_calib)
        else:
            for i in range(self.n_classes_):
                pos_probs = X[:, i]
                neg_probs = 1. - pos_probs
                bin_probs = np.stack([neg_probs, pos_probs], axis=-1)
                bin_labels = (y == i).astype(np.int32)
                bin_calib = sklearn.base.clone(self.binary_calibrator)
                bin_calib.fit(bin_probs, bin_labels)
                self.bin_calibs_.append(bin_calib)

    def predict_proba(self, X):
        X[np.isnan(X)] = 0.0
        if self.n_classes_ == 2:
            # binary classification
            return self.bin_calibs_[0].predict_proba(X)
        else:
            multi_probs = []
            for i in range(self.n_classes_):
                pos_probs = X[:, i]
                neg_probs = 1. - pos_probs
                bin_probs = np.stack([neg_probs, pos_probs], axis=-1)
                pred_probs = self.bin_calibs_[i].predict_proba(bin_probs)
                multi_probs.append(pred_probs[:, 1])
            multi_probs = np.stack(multi_probs, axis=-1)
            multi_probs = np.clip(multi_probs, a_min=1e-30, a_max=np.inf)
            multi_probs = multi_probs / np.sum(multi_probs, axis=-1, keepdims=True)
            return multi_probs


class IdentityEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        # somehow it can fail if we don't do the np.asarray()
        self.classes_ = np.asarray(list(range(n_classes)))

    def fit(self, X, y):
        # self.classes_ = np.unique(y)  # if we do this we have a problem for missing classes
        return self

    def predict_proba(self, X):
        return X

    def predict(self, X):
        # having this as a dummy here for the FrozenEstimator solution
        # which hovewer doesn't work right now
        raise NotImplementedError()


class PreprocessingCalibratorWrapper(Calibrator):
    def __init__(self, calibrator: Calibrator, pre_calib: Optional[Calibrator] = None, l2_normalize: bool = False):
        super().__init__()
        self.calibrator = calibrator
        self.pre_calib = pre_calib
        self.l2_normalize = l2_normalize

        # options: robust scaling
        # pre-TS

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor) -> 'Calibrator':
        if self.pre_calib:
            self.pre_calib_: Calibrator = sklearn.base.clone(self.pre_calib)
            self.pre_calib_.fit_torch(y_pred, y_true_labels)
            y_pred = self.pre_calib_.predict_proba_torch(y_pred)
        if self.l2_normalize:
            logits = y_pred.get_logits()
            logits = logits - logits.mean(dim=-1, keepdim=True)
            self.factor_ = 1. / (logits.square().mean().sqrt().item() + 1e-8)
            logits = self.factor_ * logits
            y_pred = CategoricalLogits(logits)

        self.calib_: Calibrator = sklearn.base.clone(self.calibrator)
        self.calib_.fit_torch(y_pred, y_true_labels)

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        if self.pre_calib:
            y_pred = self.pre_calib_.predict_proba_torch(y_pred)
        if self.l2_normalize:
            logits = y_pred.get_logits()
            logits = logits - logits.mean(dim=-1, keepdim=True)
            logits = self.factor_ * logits
            y_pred = CategoricalLogits(logits)

        return self.calib_.predict_proba_torch(y_pred)




class SklearnCalibrator(Calibrator):
    def __init__(self, method: str, cv: str):
        super().__init__()
        self.method = method
        self.cv = cv

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim == 1:
            X = np.stack((1.0-X, X), axis=1)

        n_classes = X.shape[-1]
        est = IdentityEstimator(n_classes)
        est.fit(X, y)

        # tried this workaround for deprecation warnings, but it complains in case of missing classes because it still tries to fit the estimator
        # try:
        #     # cv='prefit' option is deprecated from sklearn 1.6, FrozenEstimator was introduced in sklearn 1.6
        #     from sklearn.frozen import FrozenEstimator
        #     self.calib_ = CalibratedClassifierCV(FrozenEstimator(est), method=self.method, cv=[(np.arange(0), np.arange(X.shape[0]))])
        # except ImportError:
        #     self.calib_ = CalibratedClassifierCV(est, method=self.method, cv=self.cv)

        self.calib_ = CalibratedClassifierCV(est, method=self.method, cv=self.cv)
        self.calib_.fit(X, y)
        self.classes_ = est.classes_

    def predict_proba(self, X):
        if X.ndim == 1:
            X = np.stack((1.0-X, X), axis=1)

        return self.calib_.predict_proba(X)


class WrapCalibratorAsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, calib: BaseEstimator):
        self.calib = calib

    def fit(self, X, y):
        # print(f'Fitting calibrator: {self.calib}')
        self.calib_ = sklearn.base.clone(self.calib)
        self.calib_.fit(X, y)
        self.classes_ = list(range(np.max(y) + 1))
        return self

    def predict_proba(self, X):
        return self.calib_.predict_proba(X)


def logloss_np(y_true: np.ndarray, y_proba: np.ndarray):
    return -np.mean(np.take_along_axis(np.log(y_proba), y_true[:, None], axis=1))


class DirichletCalibrator(Calibrator):
    def __init__(self, n_cv: int = 0, use_odir: bool = False,
                 reg_lambda: float = 0.0, reg_mu: Optional[float] = None,
                 reg_lambda_grid: Optional[List[float]] = None, reg_mu_grid: Optional[List[float]] = None):
        super().__init__()

        self.n_cv = n_cv
        self.use_odir = use_odir
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu
        self.reg_lambda_grid = reg_lambda_grid
        self.reg_mu_grid = reg_mu_grid

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
        if self.n_cv == 0:
            self.cal_ = FullDirichletCalibrator(reg_lambda=self.reg_lambda, reg_mu=self.reg_mu)
        elif self.n_cv >= 2:
            reg_lambda_grid = self.reg_lambda_grid or [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
            reg_mu_grid = self.reg_mu_grid or [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
            calibrator = FullDirichletCalibrator(reg_lambda=reg_lambda_grid,
                                                 reg_mu=reg_mu_grid if self.use_odir else None)
            calibrator = WrapCalibratorAsClassifier(calibrator)
            skf = StratifiedKFold(n_splits=self.n_cv, shuffle=True, random_state=0)
            self.cal_ = GridSearchCV(calibrator, param_grid={'calib__reg_lambda': reg_lambda_grid,
                                                             'calib__reg_mu': reg_mu_grid if self.use_odir else [None]},
                                     cv=skf,
                                     # use this because scikit-learn's logloss scorer fails in case of missing classes
                                     scoring=lambda est, X_test, y_test: -logloss_np(y_test, est.predict_proba(X_test)))
        else:
            raise ValueError(f'n_cv must be either 0 or >=2, but is {self.n_cv}')

        self.cal_.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.cal_.predict_proba(X)


class MixtureCalibrator(Calibrator):
    def __init__(self, calibrator: Calibrator, output_constant: float = 1.0, input_constant: float = 0.0):
        super().__init__()
        self.calibrator = calibrator
        self.output_constant = output_constant
        self.input_constant = input_constant

    def _get_mixture(self, dist_1: CategoricalDistribution, unif_coef: float):
        probs = dist_1.get_probs()
        unif_probs = (1.0 / probs.shape[-1]) * torch.ones_like(probs)
        return CategoricalProbs((1.0 - unif_coef) * probs + unif_coef * unif_probs)

    def _fit_torch_impl(self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor):
        self.n_samples_ = y_pred.get_n_samples()
        if self.input_constant != 0.0:
            y_pred = self._get_mixture(y_pred, self.input_constant / (self.n_samples_ + 1))
        self.cal_ = sklearn.base.clone(self.calibrator)
        self.cal_.fit_torch(y_pred, y_true_labels)

    def predict_proba_torch(self, y_pred: CategoricalDistribution) -> CategoricalDistribution:
        if self.input_constant != 0.0:
            y_pred = self._get_mixture(y_pred, self.input_constant / (self.n_samples_ + 1))
        y_pred = self.cal_.predict_proba_torch(y_pred)
        if self.output_constant != 0.0:
            y_pred = self._get_mixture(y_pred, self.output_constant / (self.n_samples_ + 1))

        return y_pred


def get_calibrator(calibration_method: str, calibrate_with_mixture: bool = False,
                   cal_mixture_output_constant: float = 1.0,
                   cal_mixture_input_constant: float = 0.0, **config) -> Calibrator:
    if calibration_method == 'platt':
        cal = SklearnCalibrator(method='sigmoid', cv='prefit')
    elif calibration_method == 'platt-logits':
        cal = ApplyToLogitsCalibrator(SklearnCalibrator(method='sigmoid', cv='prefit'))
    elif calibration_method == 'isotonic':
        cal = SklearnCalibrator(method='isotonic', cv='prefit')
    elif calibration_method == 'ivap':
        cal = VennAbersCalibrator(use_ovo=config.get('va_use_ovo', False))
    elif calibration_method == 'ivap-ovr':
        cal = MulticlassOneVsRestCalibrator(BinaryVennAbersCalibrator())
    elif calibration_method == 'ivap-ovo':
        cal = MulticlassOneVsOneCalibrator(BinaryVennAbersCalibrator())
    elif calibration_method == 'isotonic-naive-ovr':
        # should be the same as 'isotonic', this is just to check
        cal = MulticlassOneVsRestCalibrator(SklearnCalibrator(method='isotonic', cv='prefit'))
    elif calibration_method == 'cir':
        cal = MulticlassOneVsRestCalibrator(CenteredIsotonicRegressionCalibrator())
    elif calibration_method in ['temp-scaling', 'ts-mix']:
        cal = TemperatureScalingCalibrator(opt=config.get('ts_opt', 'bisection'),
                                           max_bisection_steps=config.get('ts_max_bisection_steps', 30),
                                           lr=config.get('ts_lr', 0.1),
                                           max_iter=config.get('ts_max_iter', 200),
                                           use_inv_temp=config.get('ts_use_inv_temp', True),
                                           inv_temp_init=config.get('ts_inv_temp_init', 1 / 1.5))
    elif calibration_method == 'linear-scaling':
        cal = BinaryLogisticCalibrator(type='linear')
    elif calibration_method == 'affine-scaling':
        cal = BinaryLogisticCalibrator(type='affine')
    elif calibration_method == 'quadratic-scaling':
        cal = BinaryLogisticCalibrator(type='quadratic')
    elif calibration_method == 'svs':
        cal = SVSCalibrator(penalty=config.get('svs_penalty', 'ridge'),
                            rho=config.get('svs_rho', 1.0),
                            tau=config.get('svs_tau', 1.0),
                            lambda_intercept=config.get('svs_lambda_intercept', 1.0),
                            lambda_diagonal=config.get('svs_lambda_diagonal', 1.0),
                            max_iter=config.get('svs_max_iter', 500),
                            tol=config.get('svs_tol', 1e-5),
                            opt=config.get('svs_opt', 'saga'),
                            ts_preprocessing=config.get('svs_ts_preprocessing', True),
                            print_init_info=config.get('svs_print_init_info', True))
    elif calibration_method == 'sms':
        cal = SMSCalibrator(penalty=config.get('sms_penalty', 'ridge'),
                            rho=config.get('sms_rho', 1.0),
                            tau=config.get('sms_tau', 1.0),
                            lambda_intercept=config.get('sms_lambda_intercept', 1.0),
                            lambda_diagonal=config.get('sms_lambda_diagonal', 1.0),
                            lambda_off_diagonal=config.get('sms_lambda_off_diagonal', 1.0),
                            max_iter=config.get('sms_max_iter', 500),
                            tol=config.get('sms_tol', 1e-5),
                            opt=config.get('sms_opt', 'saga'),
                            ts_preprocessing=config.get('sms_ts_preprocessing', True),
                            print_init_info=config.get('sms_print_init_info', True))
    elif calibration_method == 'logistic':
        cal = LogisticCalibrator(binary_type=config.get('logistic_binary_type', 'affine'),
                                 multiclass_type=config.get('logistic_multiclass_type', 'sms'))
    elif calibration_method == 'torchunc-ts':
        cal = TorchUncertaintyTemperatureScalingCalibrator(init_val=config.get('ts_init_val', 1),
                                                           lr=config.get('ts_lr', 0.1),
                                                           max_iter=config.get('ts_max_iter', 100))
    elif calibration_method == 'guo-ts':
        cal = GuoTemperatureScalingCalibrator()
    elif calibration_method == 'torchcal-ts':
        cal = TorchcalTemperatureScalingCalibrator()
    elif calibration_method == 'autogluon-ts':
        cal = AutoGluonTemperatureScalingCalibrator(init_val=config.get('ts_init_val', 1),
                                                    max_iter=config.get('ts_max_iter', 200),
                                                    lr=config.get('ts_lr', 0.1))
    elif calibration_method == 'autogluon-inv-ts':
        cal = AutoGluonInverseTemperatureScalingCalibrator(init_val=config.get('ts_init_val', 1),
                                                           max_iter=config.get('ts_max_iter', 200),
                                                           lr=config.get('ts_lr', 0.1))
    elif calibration_method == 'dircal':
        cal = DirichletCalibrator(n_cv=0, reg_lambda=config.get('dircal_reg_lambda', 0.0),
                                  reg_mu=config.get('dircal_reg_mu', None))
    elif calibration_method == 'dircal-cv':
        cal = DirichletCalibrator(n_cv=config.get('dircal_n_cv', 5), use_odir=config.get('dircal_use_odir', False),
                                  reg_lambda_grid=config.get('dircal_reg_lambda_grid', None),
                                  reg_mu_grid=config.get('dircal_reg_mu_grid', None))
    else:
        raise ValueError(f'Unknown calibration method "{calibration_method}"')

    if calibrate_with_mixture or calibration_method.endswith('-mix'):
        cal = MixtureCalibrator(cal, output_constant=cal_mixture_output_constant,
                                input_constant=cal_mixture_input_constant)

    return cal
