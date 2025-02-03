from typing import Tuple, List

import tests  # otherwise coverage warns

import pytest
import sklearn
import numpy as np
import scipy
import torch.autograd.graph
import torchmetrics
from sklearn.model_selection import train_test_split

from probmetrics.calibrators import Calibrator, VennAbersCalibrator, MulticlassOneVsRestCalibrator, \
    BinaryVennAbersCalibrator, MulticlassOneVsOneCalibrator, SklearnCalibrator, get_calibrator
from probmetrics.distributions import CategoricalDirac, CategoricalProbs
from probmetrics.metrics import BrierLoss, Metrics, MetricsWithCalibration, Metric, AUROCOneVsRestSklearn, \
    AUROCOneVsRest, Accuracy, TorchmetricsClassificationMetric, LogLoss, ClippedLogLoss
from probmetrics.splitters import CVSplitter, AllSplitter

from test_calibrators import calib_dataset


# possible tests:
# test metrics vs sklearn or torchmetrics implementations
# test that the returned names are the same as get_names() etc.
# test missing classes / small dataset sizes?


@pytest.mark.parametrize('metrics', [
    [AUROCOneVsRestSklearn(), AUROCOneVsRest()],
    [Accuracy(), TorchmetricsClassificationMetric('accuracy_torchmetrics', False, torchmetrics.Accuracy)],
    [LogLoss(), ClippedLogLoss(clip_threshold=0.0)]
])
def test_metrics_equal(metrics: List[Metric], calib_dataset: Tuple[np.ndarray, np.ndarray]):
    X, y = calib_dataset
    n_classes = X.shape[-1]
    y_true = CategoricalDirac(torch.as_tensor(y), n_classes=n_classes)
    y_pred = CategoricalProbs(torch.as_tensor(X))

    results = [metric.compute(y_true, y_pred) for metric in metrics]

    for i in range(1, len(results)):
        np.testing.assert_allclose(results[i], results[0], atol=1e-7)


@pytest.mark.parametrize('metrics', [
    Metrics.from_names(
        ['logloss', 'brier', 'accuracy', 'class-error', 'auroc-ovr', 'auroc-ovr-sklearn', 'auroc-ovo-sklearn',
         'logloss-clip1e-06', 'smece', 'ece-15', 'rmsce-15', 'mce-15',
         'mpn_logloss', 'mpn_brier', 'mpn_accuracy']),
    MetricsWithCalibration(Metrics.from_names(['logloss', 'brier', 'accuracy', 'class-error']),
                           calibrator=get_calibrator('temp-scaling'), val_splitter=CVSplitter(n_cv=5)),
    MetricsWithCalibration(Metrics.from_names(['logloss', 'brier', 'accuracy', 'class-error']),
                           calibrator=get_calibrator('temp-scaling'), val_splitter=AllSplitter()),
])
def test_metrics_names(metrics: Metrics, calib_dataset: Tuple[np.ndarray, np.ndarray]):
    X, y = calib_dataset
    n_classes = X.shape[-1]
    y_true = CategoricalDirac(torch.as_tensor(y), n_classes=n_classes)
    y_pred = CategoricalProbs(torch.as_tensor(X))

    results = metrics.compute_all(y_true, y_pred)

    assert set(results.keys()) == set(metrics.get_names())
