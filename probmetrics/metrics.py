from collections import Counter
from typing import Optional, Dict, List, Callable, Literal, Union

import sklearn
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from probmetrics import utils
from probmetrics.calibrators import Calibrator, get_calibrator
from probmetrics.distributions import Distribution, CategoricalDistribution, ContinuousDistribution, CategoricalProbs, \
    CategoricalDirac, CategoricalLogits
from probmetrics.splitters import Splitter, CVSplitter, AllSplitter
from probmetrics.torch_utils import remove_missing_classes
from probmetrics.classifiers import WS_CatboostClassifier


class MetricType:
    CLASS = 'class'
    REG = 'reg'


# want multimetrics for Brier/Logloss on top of calibrator?
# but we need them as single metrics at least for xgboost callbacks
# maybe one can write them as metrics of transformed inputs


# could have a class hierarchy
# sample-wise metric
# classification/regression?
# Sklearn metric / torchmetrics metric
# is the metric differentiable? (not the case for sklearn)
# possibility to return multiple metrics?  (e.g. for logloss and n_logloss, accuracy and class_error, or so)

# todo: what about indicating that a metric is only for binary classification?


class Metrics:
    # todo: type-hinting for enum type?
    def __init__(self, names: List[str], is_lower_better_list: List[bool], metric_type: str, requires_f_x: bool = False,
                 binary_as_multiclass: bool = True):
        assert len(names) == len(is_lower_better_list)
        self.names = names
        self.is_lower_better_list = is_lower_better_list
        self.metric_type = metric_type
        self.requires_f_x = requires_f_x
        self.binary_as_multiclass = binary_as_multiclass

    def get_metric_type(self) -> str:
        return self.metric_type

    def get_names(self) -> List[str]:
        return self.names

    def compute_all(self, y_true: Distribution, y_pred: Distribution,
                    other_metric_values: Optional[Dict[str, torch.Tensor]] = None, reduction: str = 'mean',
                    weights: Optional[torch.Tensor] = None, f_x: Optional[Distribution] = None) -> Dict[str, torch.Tensor]:
        """
        Compute the metrics values. Should be overridden by subclasses.

        :param y_true: True labels / label distributions.
        :param y_pred: Predicted distribution.
        :param other_metric_values: Optionally, values of other metrics computed for the same (y_true, y_pred).
            If possible, a metrics object can use these values to save some re-computation.
            Can only be used if reduction == 'mean' and weights is None.
        :param reduction: Whether the metric should produce one value ('mean') or one value per sample ('none').
        Note that the option 'none' is not supported by all subclasses.
        :param weights: (Optional) sample weights of shape (n_samples,). Can only be used with 'mean' reduction.
        Weights are not implemented by all subclasses.
        :param f_x: Values f(x) for calibration-related proper losses (these can be sample-dependent).
        :return:
        """
        raise NotImplementedError()

    def compute_all_from_labels_probs(self, y_true: torch.Tensor, y_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute metrics given torch tensors.

        :param y_true: Labels with shape (n_samples,)
        :param y_probs: Probabilities with shape (n_samples, n_classes)
        :return: Dict with (metric_name, metric_value) pairs
        """
        # todo: this should be in a ClassificationMetrics subclass?
        assert self.metric_type == MetricType.CLASS
        return self.compute_all(CategoricalDirac(y_true, n_classes=y_probs.shape[-1]), CategoricalProbs(y_probs))

    def compute_all_from_labels_logits(self, y_true: torch.Tensor, y_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute metrics given torch tensors.

        :param y_true: Labels with shape (n_samples,)
        :param y_logits: Logits with shape (n_samples, n_classes)
        :return: Dict with (metric_name, metric_value) pairs
        """
        assert self.metric_type == MetricType.CLASS
        return self.compute_all(CategoricalDirac(y_true, n_classes=y_logits.shape[-1]), CategoricalLogits(y_logits))

    @staticmethod
    def _get_candidate_metrics() -> List['Metrics']:
        proper_metrics = CombinedMetrics([LogLoss(), BrierLoss(), ClippedLogLoss(),
                                          ProperLpLoss(p=1), ProperLpLoss(p=2), ProperLpLoss(p=float('inf'))])
        cand_list = [ClassError(), Accuracy(), LogLoss(), BrierLoss(), ClippedLogLoss(), AUROCOneVsRest(),
                     AUROCOneVsOneSklearn(), AUROCOneVsRestSklearn(), OneMinusAUROCOneVsRest(),
                     CalibrationError(norm='l1'), CalibrationError(norm='l2'), CalibrationError(norm='max'),
                     ProperLpLoss(p=1), ProperLpLoss(p=2), ProperLpLoss(p=float('inf')),
                     ProperLpLoss(p=1, binary_as_multiclass=True),
                     ProperLpLoss(p=2, binary_as_multiclass=True),
                     ProperLpLoss(p=float('inf'), binary_as_multiclass=True),
                     TopClassLoss(UnderConfidenceLoss(ProperLpLoss(p=1))),
                     TopClassLoss(OverConfidenceLoss(ProperLpLoss(p=1))),
                     TopClassLoss(ProperLpLoss(p=1)),
                     TopClassLoss(ProperLpLoss(p=2)),
                     TopClassLoss(ProperLpLoss(p=float('inf'))),
                     BrierLoss(binary_as_multiclass=False),
                     SmoothCalibrationError(),
                     MSE(), RMSE(), NRMSE(), MAE(), NMAE()]
        cand_list.extend([MeanProbNormalizedMetric(metric) for metric in
                          [ClippedLogLoss(), LogLoss(), BrierLoss(), ClassError(), Accuracy()]])
        for splitter in [CVSplitter(n_cv=5), AllSplitter()]:
            cand_list.extend([
                MetricsWithCalibration(proper_metrics,
                                       get_calibrator('temp-scaling', calibrate_with_mixture=True),
                                       val_splitter=splitter, cal_name='ts-mix',
                                       random_state=0),
                MetricsWithCalibration(proper_metrics,
                                       get_calibrator('isotonic', calibrate_with_mixture=True),
                                       val_splitter=splitter, cal_name='isotonic-mix',
                                       random_state=0),
                MetricsWithCalibration(proper_metrics,
                                       WS_CatboostClassifier(),
                                       val_splitter=splitter,
                                       random_state=0),
            ])

        return cand_list

    @staticmethod
    def get_available_names(metric_type: str) -> List[str]:
        return sum([m.get_names() for m in Metrics._get_candidate_metrics() if m.get_metric_type() == metric_type], [])

    @staticmethod
    def from_names(metric_names: List[str]) -> 'Metrics':
        available_metrics_list = Metrics._get_candidate_metrics()

        # print(f'Available metric names: {sum([m.get_names() for m in available_metrics_list], [])}')

        used_metrics_list = []

        for metric_name in metric_names:
            if not any([metric_name in m.get_names() for m in used_metrics_list]):
                # need a new metric
                for metrics in available_metrics_list:
                    if metric_name in metrics.get_names():
                        used_metrics_list.append(metrics)
                        break
                else:
                    # check for parametrized metrics like quantile etc.
                    if metric_name.startswith('ece-'):
                        used_metrics_list.append(CalibrationError(n_bins=int(metric_name.lstrip('ece-')), norm='l1'))
                    elif metric_name.startswith('rmsce-'):
                        used_metrics_list.append(CalibrationError(n_bins=int(metric_name.lstrip('rmsce-')), norm='l2'))
                    elif metric_name.startswith('mce-'):
                        used_metrics_list.append(CalibrationError(n_bins=int(metric_name.lstrip('mce-')), norm='max'))
                    else:
                        # no metric found
                        raise ValueError(f'Unknown metric "{metric_name}"')

        return CombinedMetrics(used_metrics_list, metric_names)


class CombinedMetrics(Metrics):
    """
    Append the outputs of multiple Metrics objects, and possibly subselect them.
    """

    def __init__(self, metrics_list: List[Metrics], used_metric_names: Optional[List[str]] = None):
        assert len(metrics_list) > 0
        assert all([m.metric_type == metrics_list[0].metric_type for m in metrics_list])
        names = sum([m.get_names() for m in metrics_list], [])
        is_lower_better = sum([m.is_lower_better_list for m in metrics_list], [])
        if used_metric_names is not None:
            assert all([name in names for name in used_metric_names])
            zip_dict = {n: i for n, i in zip(names, is_lower_better)}
            names = used_metric_names
            is_lower_better = [zip_dict[name] for name in names]

        # check for duplicates in the names
        counts = Counter(names)
        duplicates = [item for item, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Multiple metrics generate results with the same name, duplicate names are: {duplicates}")

        super().__init__(names=names,
                         is_lower_better_list=is_lower_better,
                         metric_type=metrics_list[0].metric_type
                         )
        self.metrics_list = metrics_list

    def compute_all(self, y_true: Distribution, y_pred: Distribution,
                    other_metric_values: Optional[Dict[str, torch.Tensor]] = None, reduction: str = 'mean',
                    weights: Optional[torch.Tensor] = None, f_x: Optional[Distribution] = None) -> Dict[str, torch.Tensor]:
        metric_values = dict()
        other_metric_values = dict() if other_metric_values is None else other_metric_values

        for metrics in self.metrics_list:
            results = metrics.compute_all(y_true, y_pred, metric_values, reduction, weights, f_x=f_x)
            metric_values = utils.join_dicts(metric_values, results, allow_overlap=False)
            other_metric_values = utils.join_dicts(other_metric_values, results)  # here we can have overlap

        return {key: value for key, value in metric_values.items() if key in self.get_names()}


class Metric(Metrics):
    def __init__(self, name: str, is_lower_better: bool, metric_type: str, requires_f_x: bool = False, binary_as_multiclass:bool = True):
        name = name if binary_as_multiclass else name+"-binary-as-1d"
        super().__init__(names=[name], is_lower_better_list=[is_lower_better], metric_type=metric_type, requires_f_x=requires_f_x, binary_as_multiclass=binary_as_multiclass)
        self.name = name
        self.is_lower_better = is_lower_better

    def get_name(self) -> str:
        return self.get_names()[0]

    def get_is_lower_better(self) -> bool:
        return self.is_lower_better_list[0]

    def compute_all(self, y_true: Distribution, y_pred: Distribution,
                    other_metric_values: Optional[Dict[str, torch.Tensor]] = None, reduction: str = 'mean',
                    weights: Optional[torch.Tensor] = None, f_x: Optional[Distribution] = None) -> Dict[str, torch.Tensor]:
        return {self.get_name(): self.compute(y_true, y_pred, other_metric_values, reduction, weights, f_x=f_x)}

    # may have parameters  (e.g., quantile_alpha, or bin distribution for regression-as-classification)
    # alternative computations?
    # reduction?
    # batched computation?
    def compute(self, y_true: Distribution, y_pred: Distribution,
                other_metric_values: Optional[Dict[str, torch.Tensor]] = None, reduction: str = 'mean',
                weights: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # maybe use kwargs?
        # or have other functions for compute() and compute with other metric values?
        # maybe with the reduction we have the possibility to compute confidence intervals as well?
        raise NotImplementedError()

    def compute_from_labels_probs(self, y_true: torch.Tensor, y_probs: torch.Tensor) -> torch.Tensor:
        assert self.metric_type == MetricType.CLASS
        return self.compute(CategoricalDirac(y_true, n_classes=y_probs.shape[-1]), CategoricalProbs(y_probs))

    def compute_from_labels_logits(self, y_true: torch.Tensor, y_logits: torch.Tensor) -> torch.Tensor:
        assert self.metric_type == MetricType.CLASS
        return self.compute(CategoricalDirac(y_true, n_classes=y_logits.shape[-1]), CategoricalLogits(y_logits))

    @staticmethod
    def from_name(metric_name: str):
        return SelectMetric(Metrics.from_names([metric_name]), metric_name)


class SelectMetric(Metric):
    """
    Turn a Metrics object into a Metric by selecting one of its outputs.
    """

    def __init__(self, metrics: Metrics, name: str):
        super().__init__(name, metrics.is_lower_better_list[metrics.get_names().index(name)], metrics.get_metric_type())
        self.metrics = metrics

    def compute(self, y_true: Distribution, y_pred: Distribution,
                other_metric_values: Optional[Dict[str, torch.Tensor]] = None, reduction: str = 'mean',
                weights: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return self.metrics.compute_all(y_true, y_pred, other_metric_values, reduction, weights, **kwargs)[self.name]


class ClassificationMetric(Metric):
    def __init__(self, name: str, is_lower_better: bool, requires_f_x: bool = False, binary_as_multiclass: bool = True):
        super().__init__(name=name, is_lower_better=is_lower_better, metric_type=MetricType.CLASS,
                         requires_f_x=requires_f_x, binary_as_multiclass=binary_as_multiclass)

    def compute(self, y_true: Distribution, y_pred: Distribution,
                other_metric_values: Optional[Dict[str, torch.Tensor]] = None, reduction: str = 'mean',
                weights: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if not isinstance(y_pred, CategoricalDistribution):
            raise ValueError(f'Classification metrics need a categorical distribution, but got {type(y_pred)=}')
        if not isinstance(y_true, CategoricalDistribution):
            raise ValueError(f'Classification metrics need a categorical distribution, but got {type(y_true)=}')
        if y_true.get_n_classes() != y_pred.get_n_classes():
            raise ValueError(
                f"Number of classes don't match, got {y_true.get_n_classes()=} and {y_pred.get_n_classes()=}")
        if reduction == 'mean' and weights is None:
            if other_metric_values is not None and self.name in other_metric_values:
                return other_metric_values[self.name]
            if other_metric_values is not None:
                result = self._try_use_cached(y_true, y_pred, other_metric_values)
                if result is not None:
                    return result

            result = self._compute_mean(y_true, y_pred, **kwargs)
            if result is None:
                raise NotImplementedError()

            return result
        else:
            result = self._compute_indiv(y_true, y_pred, **kwargs)
            if result is None:
                raise NotImplementedError()
            if reduction == 'mean':
                assert weights is not None
                weights = weights / weights.sum(dim=-1, keepdim=True)
                return torch.dot(result, weights)
            elif reduction == 'none':
                return result
            else:
                raise ValueError(f'Unknown reduction "{reduction}"')

    def _try_use_cached(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution,
                        other_metric_values: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute the metric value using results from other_metric_values, if possible.
        Override in subclasses if applicable.

        :param y_true:
        :param y_pred:
        :param other_metric_values:
        :return: metric value or None, if no results from other metric values could be used.
        """
        return None

    def _compute_mean(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[torch.Tensor]:
        """
        Compute a metric value for the entire dataset. By default, this tries to use _compute_indiv().
        Override this method if the metric doesn't support _compute_indiv().
        :param y_true:
        :param y_pred:
        :return:
        """
        indiv_results = self._compute_indiv(y_true, y_pred, **kwargs)
        return None if indiv_results is None else indiv_results.mean(dim=-1)

    def _compute_indiv(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[
        torch.Tensor]:
        """
        Compute metric values per sample. Override this method if applicable, else override _compute_mean().

        :param y_true:
        :param y_pred:
        :return:
        """
        return None


class RegressionMetric(Metric):
    def __init__(self, name: str, is_lower_better: bool):
        super().__init__(name=name, is_lower_better=is_lower_better, metric_type=MetricType.REG)

    def compute(self, y_true: Distribution, y_pred: Distribution,
                other_metric_values: Optional[Dict[str, torch.Tensor]] = None, reduction: str = 'mean',
                weights: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # todo: do we really want y_true to be a distribution here?
        # todo: what about the multi-output setting?
        if not isinstance(y_pred, ContinuousDistribution):
            raise ValueError(f'Regression metrics need a continuous distribution, but got {type(y_pred)=}')
        if not isinstance(y_true, ContinuousDistribution):
            raise ValueError(f'Regression metrics need a continuous distribution, but got {type(y_true)=}')
        if reduction == 'mean' and weights is None:
            if other_metric_values is not None:
                if self.name in other_metric_values:
                    return other_metric_values[self.name]
                result = self._try_use_cached(y_true, y_pred, other_metric_values)
                if result is not None:
                    return result

            result = self._compute_mean(y_true, y_pred, **kwargs)
            if result is None:
                raise NotImplementedError()

            return result
        else:
            result = self._compute_indiv(y_true, y_pred, **kwargs)
            if result is None:
                raise NotImplementedError()
            if reduction == 'mean':
                assert weights is not None
                weights = weights / weights.sum(dim=-1, keepdim=True)
                return torch.dot(result, weights)
            elif reduction == 'none':
                return result
            else:
                raise ValueError(f'Unknown reduction "{reduction}"')

    def _try_use_cached(self, y_true: ContinuousDistribution, y_pred: ContinuousDistribution,
                        other_metric_values: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        return None

    def _compute_mean(self, y_true: ContinuousDistribution, y_pred: ContinuousDistribution, **kwargs) -> Optional[torch.Tensor]:
        indiv_results = self._compute_indiv(y_true, y_pred)
        return None if indiv_results is None else indiv_results.mean(dim=-1)

    def _compute_indiv(self, y_true: ContinuousDistribution, y_pred: ContinuousDistribution, **kwargs) -> Optional[
        torch.Tensor]:
        return None


# todo: normalized versions of losses?
# also loss functions for regression-as-classification (HL-Gauss etc.)
# allow to have schedulable parameters like label smoothing epsilon?

# fix balanced accuracy / mcc
# add factor 1/2 for brier score as in scikit-learn? (or vice versa)

# maybe allow to specify stop metrics together with the name of the head that they should be applied to?


# use class variables for the basic properties?


class ClassError(ClassificationMetric):
    def __init__(self):
        super().__init__(name='class-error', is_lower_better=True)

    def _try_use_cached(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution,
                        other_metric_values: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if 'accuracy' in other_metric_values:
            return 1 - other_metric_values['accuracy']
        return None

    def _compute_indiv(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[
        torch.Tensor]:
        return (y_pred.get_modes() != y_true.get_modes()).to(torch.float32)


class Accuracy(ClassificationMetric):
    def __init__(self):
        super().__init__(name='accuracy', is_lower_better=False)

    def _try_use_cached(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution,
                        other_metric_values: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if 'class-error' in other_metric_values:
            return 1 - other_metric_values['class-error']
        return None

    def _compute_indiv(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[
        torch.Tensor]:
        return (y_pred.get_modes() == y_true.get_modes()).to(torch.float32)


class LogLoss(ClassificationMetric):
    def __init__(self):
        super().__init__(name='logloss', is_lower_better=True, binary_as_multiclass=True)

    def _compute_indiv(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[
        torch.Tensor]:
        logits = y_pred.get_logits()
        if not self.binary_as_multiclass:
            raise NotImplementedError("This class does not suppoer binary_as_multiclass yet.")
        if y_true.is_dirac():
            return -F.log_softmax(logits, dim=-1).gather(-1, y_true.get_modes().unsqueeze(-1)).squeeze(-1)
        else:
            return (-F.log_softmax(logits, dim=-1) * y_true.get_probs()).sum(dim=-1)


class ClippedLogLoss(ClassificationMetric):
    def __init__(self, clip_threshold: float = 1e-6):
        super().__init__(name=f'logloss-clip{clip_threshold:g}', is_lower_better=True, binary_as_multiclass=True)
        self.clip_threshold = clip_threshold

    def _compute_indiv(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[
        torch.Tensor]:
        if not self.binary_as_multiclass:
            raise NotImplementedError("This class does not suppoer binary_as_multiclass yet.")
        probs = y_pred.get_probs()
        probs = probs.clamp(min=self.clip_threshold, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        log_probs = torch.log(probs)
        if y_true.is_dirac():
            return -log_probs.gather(-1, y_true.get_modes().unsqueeze(-1)).squeeze(-1)
        else:
            return (-log_probs * y_true.get_probs()).sum(dim=-1)


class BrierLoss(ClassificationMetric):
    def __init__(self, binary_as_multiclass: bool = True):
        """
        Brier loss.
        :param binary_as_multiclass: Whether to treat binary classification as a 2-dimensional multiclass problem
            (yields twice the Brier score from 1D, which is also twice the value from scikit-learn).
        """
        super().__init__(name='brier', is_lower_better=True, binary_as_multiclass=binary_as_multiclass)

    def _compute_indiv(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[
        torch.Tensor]:
        if y_pred.get_n_classes() <= 2 and not self.binary_as_multiclass:
            return (y_pred.get_probs()[:, 1] - y_true.get_probs()[:, 1]).square().sum(dim=-1)
        return (y_pred.get_probs() - y_true.get_probs()).square().sum(dim=-1)


def _clip_p_values(p: torch.Tensor, f_x: torch.Tensor, for_over: bool = False, for_under: bool = False) -> torch.Tensor:
    """    
    Rectifies probabilities based on the reference point f_x.
    
    Logic for_under: 1_{f>0.5} * max(p,f) + 1_{f<0.5} * min(p,f) + 1_{f=0.5} * 0.5
    Logic for_over:  1_{f>0.5} * min(p,f) + 1_{f<0.5} * max(p,f) + 1_{f=0.5} * 0.5

    :param p: (Tensor) Predicted probability distribution.
    :param f_x: (Tensor) Reference probability distribution.
    :param for_over: (bool) Whether to clip for over-confidence.
    :param for_under: (bool) Whether to clip for under-confidence.
    """
    if for_over and for_under:
        raise ValueError("Both 'for_over' and 'for_under' cannot be True simultaneously.")
    if not (for_over or for_under):
        import warnings
        msg = (
            "Confidence clipping is initialized, but both 'for_over' and 'for_under' are set to False. "
            "No values will be modified. Did you mean to enable one of these flags?"
        )
        warnings.warn(msg, category=UserWarning, stacklevel=2)

    p = torch.as_tensor(p)
    f_x = torch.as_tensor(f_x)

    if p.shape[-1] != 1:
        import warnings
        msg = (
            f"Input shape {p.shape} detected. Over/underconfidence metrics are "
            "designed for binary tasks with shape (n, 1). For multi-class data, "
            "consider using 'OverConfidenceLoss( TopClassLoss(DesiredMetric) )' or "
            "OverConfidenceLoss( TopClassLoss(DesiredMetric) ) for confidence clipping."
        )
        warnings.warn(msg, UserWarning)
    
    clipped_p = torch.full_like(p, 0.5)

    if for_under:
        clipped_p = torch.where(f_x > 0.5, torch.maximum(p, f_x), clipped_p)
        clipped_p = torch.where(f_x < 0.5, torch.minimum(p, f_x), clipped_p)
        
    elif for_over:
        clipped_p = torch.where(f_x > 0.5, torch.minimum(p, f_x), clipped_p)
        clipped_p = torch.where(f_x < 0.5, torch.maximum(p, f_x), clipped_p)

    return clipped_p


class ProperLpLoss(ClassificationMetric): 
    def __init__(self, p: float = 2., binary_as_multiclass: bool = False):
        assert p >= 1
        super().__init__(name=f'proper-L{p:.10g}', is_lower_better=True, requires_f_x=True,
                         binary_as_multiclass=binary_as_multiclass)
        self.p = p

    def _compute_indiv(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[
        torch.Tensor]:
        """
        Calculates the proper loss l_{f(X)}(g_f_x, Y) for a given norm.

        Formula: l_{f(X)}(g, y) = h(g) + <subgradient, y - g>
        where h(g) = -||g - f_x||_norm
        g_f_x = y_pred
        f_x is in kwargs
        y = y_true
        """
        f_x = kwargs["f_x"].get_probs()  # values f(x) for calibration, that's where the loss is centered
        y_pred_probs = y_pred.get_probs()
        y_true_probs = y_true.get_probs()
        if y_pred.get_n_classes() == 2 and not self.binary_as_multiclass:
            y_pred_probs, y_true_probs = y_pred_probs[:, 1:], y_true_probs[:, 1:]

        g_f_x = y_pred_probs
        y = y_true_probs

        diff = g_f_x - f_x
        norm_val = torch.linalg.norm(diff, ord=self.p, dim=1, keepdim=True)
        h_p = -norm_val.squeeze()

        if self.p == float('inf'):
            abs_diff = torch.abs(diff)
            subgradient = torch.zeros_like(diff)
            indices = torch.argmax(abs_diff, dim=1, keepdim=True)
            subgradient.scatter_(1, indices, 1.0)
            subgradient = -subgradient * torch.sign(diff)
            dot_product = torch.sum(subgradient * (y - g_f_x), dim=1)
            return dot_product + h_p

        safe_norm = torch.where(norm_val == 0, torch.ones_like(norm_val), norm_val)
        grad_norm = (torch.abs(diff) ** (self.p - 1) * torch.sign(diff)) / (safe_norm ** (self.p - 1))
        subgradient = -torch.where(norm_val == 0, torch.zeros_like(grad_norm), grad_norm)
        dot_product = torch.sum(subgradient * (y - g_f_x), dim=1)

        return dot_product + h_p


class TopClassLoss(ClassificationMetric):
    def __init__(self, metric: Metrics):
        """
        Wrapper to calculate a given metric focusing only on the class 
        where f(x) has the highest confidence and treating it as binary.
        If f_x is provided, then f_x is used to determine the top class, otherwise y_pred is used.
        
        :param metric: An instantiated single Metric object.
        """
        if isinstance(metric, str):
            raise TypeError("Expected a metric object, got a string. Use TopClassWrapper.from_name() instead.")
        
        if isinstance(metric, list) or metric.__class__.__name__ == 'CombinedMetrics':
            raise ValueError("TopClassWrapper only supports a single metric object.")

        self.base_metric = metric

        base_name = self.base_metric.get_names()[0]
        is_lower_better = self.base_metric.is_lower_better_list[0]
        wrapper_name = f"topclass-{base_name}"        
        
        super().__init__(name=wrapper_name, is_lower_better=is_lower_better, requires_f_x=False)

    @classmethod
    def from_name(cls, metric_name: str) -> 'TopClassLoss':
        """
        Initialize the wrapper directly from a metric's string name.
        """
        base_metric = cls._resolve_single_metric(metric_name)
        return cls(metric=base_metric)

    @staticmethod
    def _resolve_single_metric(name: str) -> Metrics:
        """Helper to pull the specific base class out of candidates."""
        return Metrics.from_names([name]).metrics_list[0]

    def _extract_top_class(self, 
                           y_true: CategoricalDistribution, 
                           y_pred: CategoricalDistribution, 
                           f_x: CategoricalDistribution):
        """
        Slices the distributions down to just the top class predicted by f_x.
        Returns wrapped CategoricalProbs with two classes.
        """
        y_true_probs = y_true.get_probs()
        y_pred_probs = y_pred.get_probs()
        f_x_probs = f_x.get_probs()

        top_indices = torch.argmax(f_x_probs, dim=-1)
        row_indices = torch.arange(f_x_probs.size(0), device=f_x_probs.device)

        y_true_top = y_true_probs[row_indices, top_indices]
        y_pred_top = y_pred_probs[row_indices, top_indices].unsqueeze(-1)
        f_x_top = f_x_probs[row_indices, top_indices].unsqueeze(-1)
    
        y_pred_top = torch.cat([1 - y_pred_top, y_pred_top], dim=1)
        f_x_top = torch.cat([1 - f_x_top, f_x_top],  dim=1)

        return CategoricalDirac(torch.as_tensor(y_true_top).long(), n_classes=2), CategoricalProbs(y_pred_top), CategoricalProbs(f_x_top)

    def _compute_indiv(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[torch.Tensor]:
        """
        Isolates the top class distributions, updates kwargs, and defers to base metric.
        """
        f_x = kwargs.get("f_x", y_pred)  # use predictions for choosing the top class if f_x is not provided

        y_true_top, y_pred_top, f_x_top = self._extract_top_class(y_true, y_pred, f_x)
        
        updated_kwargs = kwargs.copy()
        updated_kwargs["f_x"] = f_x_top
        
        return self.base_metric._compute_indiv(y_true=y_true_top, y_pred=y_pred_top, **updated_kwargs)


class ConfidenceLoss(ClassificationMetric):
    def __init__(self, metric: Metrics, direction: str):
        """
        Wrapper to calculate under- and over-confidence for a single metric.
        
        :param metric: An instantiated single Metric object.
        :param direction: 'over' or 'under'
        """
        if isinstance(metric, str):
            raise TypeError("Expected a metric object, got a string. Use ConfidenceWrapper.from_name() to initialize from a string.")
        
        if isinstance(metric, list) or metric.__class__.__name__ == 'CombinedMetrics':
            raise ValueError("ConfidenceWrapper only supports a single metric object. Multiple metrics or CombinedMetrics are not supported.")

        assert direction in ['over', 'under'], "direction must be 'over' or 'under'"
        
        self.direction = direction
        self.base_metric = metric

        base_name = self.base_metric.get_names()[0]
        is_lower_better = self.base_metric.is_lower_better_list[0]
        wrapper_name = f"{direction}-{base_name}"
        super().__init__(name=wrapper_name, is_lower_better=is_lower_better, requires_f_x=True)

    @classmethod
    def from_name(cls, metric_name: str, direction: str) -> 'ConfidenceLoss':
        """
        Initialize the wrapper directly from a metric's string name.
        """
        base_metric = cls._resolve_single_metric(metric_name)
        return cls(metric=base_metric, direction=direction)

    @staticmethod
    def _resolve_single_metric(name: str) -> Metrics:
        """Helper to pull the specific base class out of candidates."""
        return Metrics.from_names([name]).metrics_list[0]

    def _clip_distribution(self, y_pred: CategoricalDistribution, f_x: CategoricalDistribution) -> CategoricalDistribution:
        """Helper to handle the clipping logic dynamically based on direction."""
        if self.direction == 'over':
            clipped_probs = _clip_p_values(y_pred.get_probs(), f_x.get_probs(), for_over=True)
        elif self.direction == 'under':
            clipped_probs = _clip_p_values(y_pred.get_probs(), f_x.get_probs(), for_under=True)
        else:
            raise ValueError("direction must be 'over' or 'under'")
            
        return CategoricalProbs(clipped_probs)

    def _compute_indiv(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[torch.Tensor]:
        """
        Standard individual computation for single ClassificationMetrics.
        """
        f_x = kwargs.get("f_x")
        assert f_x is not None, f"f_x is required for {self.get_names()[0]}"

        clipped_y_pred = self._clip_distribution(y_pred, f_x)
        
        return self.base_metric._compute_indiv(y_true=y_true, y_pred=clipped_y_pred, **kwargs)


class OverConfidenceLoss(ConfidenceLoss):
    def __init__(self, metric: Metrics):
        """
        Specialized wrapper for calculating over-confidence for a single metric.
        
        :param metric: An instantiated single Metric object.
        """
        # We explicitly set direction to 'over'
        super().__init__(metric=metric, direction='over')

    @classmethod
    def from_name(cls, metric_name: str) -> 'OverConfidenceLoss':
        """
        Initialize the over-confidence wrapper directly from a metric's string name.
        """
        return cls(metric=cls._resolve_single_metric(metric_name))


class UnderConfidenceLoss(ConfidenceLoss):
    def __init__(self, metric: Metrics):
        """
        Specialized wrapper for calculating under-confidence for a single metric.
        
        :param metric: An instantiated single Metric object.
        """
        # We explicitly set direction to 'under'
        super().__init__(metric=metric, direction='under')

    @classmethod
    def from_name(cls, metric_name: str) -> 'UnderConfidenceLoss':
        """
        Initialize the under-confidence wrapper directly from a metric's string name.
        """
        return cls(metric=cls._resolve_single_metric(metric_name))


class BalancedAccuracy(ClassificationMetric):
    pass  # todo: could be implemented using SklearnClassificiationMetric but this should use threshold tuning


class SklearnClassificationMetric(ClassificationMetric):
    def __init__(self, name: str, is_lower_better: bool, sklearn_func: Callable, uses_probs: bool,
                 two_class_single_column: bool):
        super().__init__(name=name, is_lower_better=is_lower_better)
        self.sklearn_func = sklearn_func
        self.uses_probs = uses_probs
        self.two_class_single_column = two_class_single_column

    def _compute_mean(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[torch.Tensor]:
        # adapted from https://github.com/dholzmueller/pytabkit/blob/39086ac0621918de315c234d4719411705e13ea1/pytabkit/models/training/metrics.py#L518
        # handle classes that don't occur in the test set
        y_probs = y_pred.get_probs()
        y = y_true.get_modes()
        y_probs, y = remove_missing_classes(y_probs, y)
        y_probs = y_probs / y_probs.sum(dim=-1, keepdim=True)

        if self.uses_probs:
            # convert logits to probabilities
            y_pred_np = y_probs.cpu().numpy()
            if y_pred_np.shape[-1] == 2 and self.two_class_single_column:
                # binary classification, scikit-learn expects only probabilities of the positive class
                y_pred_np = y_pred_np[..., 1]
        else:
            # convert logits to predicted class
            y_pred_np = torch.argmax(y_probs, dim=-1).cpu().numpy()

        y_np = y.cpu().numpy()
        return torch.as_tensor(self.sklearn_func(y_np, y_pred_np))

class AUROCOneVsRestSklearn(SklearnClassificationMetric):
    def __init__(self):
        super().__init__('auroc-ovr-sklearn', is_lower_better=False,
                         sklearn_func=lambda y1, y2: roc_auc_score(y1, y2, multi_class='ovr'), uses_probs=True,
                         two_class_single_column=True)

    # todo: could compute from ovo in the binary case and vice versa


class AUROCOneVsOneSklearn(SklearnClassificationMetric):
    def __init__(self):
        super().__init__('auroc-ovo-sklearn', is_lower_better=False,
                         sklearn_func=lambda y1, y2: roc_auc_score(y1, y2, multi_class='ovo'), uses_probs=True,
                         two_class_single_column=True)


class TorchmetricsClassificationMetric(ClassificationMetric):
    def __init__(self, name: str, is_lower_better: bool, torch_metric):
        super().__init__(name=name, is_lower_better=is_lower_better)
        self.torch_metric = torch_metric

    def _get_torch_metric(self, n_classes: int):
        # can be overridden instead of specifying torch_metric in the constructor if more control is needed
        return self.torch_metric(task='binary' if n_classes == 2 else 'multiclass', num_classes=n_classes)

    def _compute_mean(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[torch.Tensor]:
        import torchmetrics
        y_probs = y_pred.get_probs()
        y = y_true.get_modes()

        # handle classes that don't occur in the test set
        y_probs, y = remove_missing_classes(y_probs, y)

        # ensure that no probabilities are zero or one to circumvent some problems
        # https://github.com/Lightning-AI/torchmetrics/issues/1646
        y_probs = y_probs.clamp(1e-7, 1 - 1e-7)
        y_probs = y_probs / y_probs.sum(dim=-1, keepdim=True)

        num_classes = y_probs.shape[-1]
        is_binary = num_classes == 2
        if is_binary:
            # binary classification, torchmetrics expects only probabilities of the positive class
            y_probs = y_probs[..., 1]

        metric = self._get_torch_metric(n_classes=num_classes)
        return torch.as_tensor(metric.forward(y_probs, y), dtype=torch.float32)


class AUROCOneVsRest(TorchmetricsClassificationMetric):
    def __init__(self):
        import torchmetrics
        super().__init__(name='auroc-ovr', is_lower_better=False, torch_metric=torchmetrics.AUROC)


class OneMinusAUROCOneVsRest(ClassificationMetric):
    def __init__(self):
        super().__init__(name='1-auroc-ovr', is_lower_better=True)

    def _try_use_cached(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution,
                        other_metric_values: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if 'auroc-ovr' in other_metric_values:
            return 1. - other_metric_values['auroc_ovr']
        return None

    def _compute_mean(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[torch.Tensor]:
        return 1. - AUROCOneVsRest().compute(y_true=y_true, y_pred=y_pred)


class CalibrationError(TorchmetricsClassificationMetric):
    def __init__(self, n_bins: int = 15, norm: Literal['l1', 'l2', 'max'] = 'l1'):
        import torchmetrics
        if norm == 'l1':
            name = f'ece-{n_bins}'
        elif norm == 'l2':
            name = f'rmsce-{n_bins}'
        elif norm == 'max':
            name = f'mce-{n_bins}'
        else:
            raise ValueError(f'Unknown value {norm=}')
        super().__init__(name=name, is_lower_better=True, torch_metric=torchmetrics.CalibrationError)
        self.n_bins = n_bins
        self.norm = norm

    def _get_torch_metric(self, n_classes: int):
        import torchmetrics
        return torchmetrics.CalibrationError(task='binary' if n_classes == 2 else 'multiclass', num_classes=n_classes,
                                             n_bins=self.n_bins, norm=self.norm)


class SmoothCalibrationError(ClassificationMetric):
    def __init__(self):
        super().__init__(name='smece', is_lower_better=True)

    def _compute_mean(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[torch.Tensor]:
        import relplot as rp

        labels_torch = y_true.get_modes()
        labels = labels_torch.detach().cpu().numpy()
        if y_pred.get_n_classes() > 2:
            # multiclass, reduce to binary setting
            conf, acc = rp.multiclass_logits_to_confidences(y_pred.get_logits().detach().cpu().numpy(),
                                                            y_true.get_modes().detach().cpu().numpy())
        else:
            # binary
            conf = y_pred.get_probs().detach().cpu().numpy()[:, 1]
            acc = labels
        return torch.as_tensor(rp.smECE(f=conf, y=acc), device=labels_torch.device, dtype=torch.float32)


class MetricsWithCalibration(Metrics):
    def __init__(self, metrics: Metrics, calibrator: sklearn.base.ClassifierMixin, val_splitter: Splitter,
                 cal_name: Optional[str] = None, random_state: Optional[int] = None):
        if cal_name is None:
            cal_name = str(calibrator.__class__.__name__)
        name_suffix = f'_{cal_name}_{val_splitter.get_name()}'
        names = [f'{decomp_part}_{name}{name_suffix}' for decomp_part in ['calib-err', 'refinement']
                 for name in metrics.get_names()]
        super().__init__(names=names, is_lower_better_list=metrics.is_lower_better_list * 2,
                         metric_type=MetricType.CLASS)
        self.metrics = metrics
        self.calibrator = calibrator
        self.val_splitter = val_splitter
        self.random_state = random_state
        self.name_suffix = name_suffix

    def compute_all(self, y_true: Distribution, y_pred: Distribution,
                    other_metric_values: Optional[Dict[str, torch.Tensor]] = None, reduction: str = 'mean',
                    weights: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        assert isinstance(y_true, CategoricalDistribution)
        assert isinstance(y_pred, CategoricalDistribution)

        splits = self.val_splitter.get_splits(y_true=y_true, random_state=self.random_state)

        orig_probs = []
        cal_probs = []
        true_probs = []

        # is_binary = y_pred.get_probs().shape[-1] == 1

        for train_idxs, val_idxs in splits:
            cal: Calibrator = sklearn.base.clone(self.calibrator)
            if isinstance(cal, Calibrator):
                cal.fit_torch(CategoricalProbs(y_pred.get_probs()[train_idxs]), y_true.get_modes()[train_idxs])
                y_val_probs = cal.predict_proba_torch(CategoricalProbs(y_pred.get_probs()[val_idxs])).get_probs()
            else: 
                cal.fit( y_pred.get_probs().detach().cpu().numpy()[train_idxs], y_true.get_modes().detach().cpu().numpy()[train_idxs])
                y_val_probs = torch.tensor( cal.predict_proba( y_pred.get_probs().detach().cpu().numpy()[val_idxs] ) )

            # todo: fix this
            # if is_binary:
            #     orig_probs.append(y_pred.get_probs()[val_idxs])
            #     cal_probs.append(y_val_probs[:, [1]])
            #     true_probs.append(y_true.get_modes()[val_idxs].reshape(-1, 1)) 
            # else:
            orig_probs.append(y_pred.get_probs()[val_idxs])
            cal_probs.append(y_val_probs)
            true_probs.append(y_true.get_probs()[val_idxs])

        y_pred_orig_dist = CategoricalProbs(torch.cat(orig_probs, dim=-2))
        y_pred_cal_dist = CategoricalProbs(torch.cat(cal_probs, dim=-2))
        y_true_dist = CategoricalProbs(torch.cat(true_probs, dim=-2))

        orig_results = self.metrics.compute_all(y_true=y_true_dist, y_pred=y_pred_orig_dist, f_x=y_pred_orig_dist,
                                                other_metric_values=other_metric_values)
        ref_results = self.metrics.compute_all(y_true=y_true_dist, y_pred=y_pred_cal_dist, f_x=y_pred_orig_dist)
    
        calerr_results = {f'calib-err_{key}{self.name_suffix}': orig_results[key] - ref_results[key] for key in
                          orig_results}
        ref_results = {f'refinement_{key}{self.name_suffix}': value for key, value in ref_results.items()}

        return {**calerr_results, **ref_results}

        # todo: provide individual results if the splitter allows it?  (maybe add a function in the splitter to check?)


class MeanProbNormalizedMetric(ClassificationMetric):
    def __init__(self, metric: ClassificationMetric):
        super().__init__(name='mpn_' + metric.name, is_lower_better=metric.is_lower_better)
        self.metric = metric

    def _try_use_cached(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution,
                        other_metric_values: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.metric.name in other_metric_values:
            unnorm_value = other_metric_values[self.metric.name]
            ref_value = self.metric.compute(y_true, CategoricalProbs(
                y_pred.get_probs().mean(dim=0, keepdim=True).expand(y_pred.get_n_samples(), -1)))

            return unnorm_value / (ref_value + 1e-30)
        else:
            return None

    def _compute_mean(self, y_true: CategoricalDistribution, y_pred: CategoricalDistribution, **kwargs) -> Optional[torch.Tensor]:
        unnorm_value = self.metric.compute(y_true, y_pred)
        ref_value = self.metric.compute(y_true, CategoricalProbs(
            y_pred.get_probs().mean(dim=0, keepdim=True).expand(y_pred.get_n_samples(), -1)))

        return unnorm_value / (ref_value + 1e-30)


# doubly-smoothed log-loss?


# calibration error metrics

# normalized error metrics

# maybe spherical loss


# ----- regression metrics -----

class MSE(RegressionMetric):
    def __init__(self):
        super().__init__(name='mse', is_lower_better=True)

    pass


class RMSE(RegressionMetric):
    def __init__(self):
        super().__init__(name='rmse', is_lower_better=True)

    pass


class NRMSE(RegressionMetric):
    def __init__(self):
        super().__init__(name='nrmse', is_lower_better=True)

    pass


class MAE(RegressionMetric):
    def __init__(self):
        super().__init__(name='mae', is_lower_better=True)

    pass


class NMAE(RegressionMetric):
    def __init__(self):
        super().__init__(name='nmae', is_lower_better=True)

    pass


class PinballLoss(RegressionMetric):
    def __init__(self):
        super().__init__(name='pinball', is_lower_better=True)  # todo: add parameter

    pass


class NormalizedPinballLoss(RegressionMetric):
    pass


class CRPS(RegressionMetric):
    pass

# max error
# coverage?
# CRPS
# regression-as-classification
