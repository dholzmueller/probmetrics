from typing import Tuple, Optional

import tests  # otherwise coverage warns

import pytest
import sklearn
import numpy as np
import scipy
import torch.autograd.graph
from sklearn.model_selection import train_test_split

from probmetrics.calibrators import Calibrator, VennAbersCalibrator, MulticlassOneVsRestCalibrator, \
    BinaryVennAbersCalibrator, MulticlassOneVsOneCalibrator, SklearnCalibrator, get_calibrator, \
    TemperatureScalingCalibrator
from probmetrics.distributions import CategoricalDirac, CategoricalProbs
from probmetrics.metrics import BrierLoss, ClassificationMetric, SmoothCalibrationError, CalibrationError


def sample_labels(p: np.ndarray, random_state: Optional[int] = None) -> np.ndarray:
    """
    Sample labels according to probabilities in p.
    :param p: Vector of probabilities, shape is (n_samples, n_classes).
    :param random_state: random seed
    :return:
    """
    # from ChatGPT
    # Assuming prob_array is your (n_samples, n_classes) numpy array
    # Each row in prob_array is a probability distribution (it sums to 1)
    rng = np.random.default_rng(seed=random_state)
    cumulative_probs = np.cumsum(p, axis=-1)  # Cumulative sum along each row
    random_values = rng.random(size=(*p.shape[:-1], 1))  # Uniform random values for each sample
    # print(f'{p.shape=}, {cumulative_probs.shape=}, {random_values.shape=}')
    samples = (random_values < cumulative_probs).argmax(
        axis=-1)  # Find the first index where cumulative probability exceeds random value
    # print(f'{samples.shape=}')
    return samples


@pytest.fixture(params=[(500, 2, 0.3, 0), (2000, 4, 2.0, 1)])
# def calib_dataset(n_samples: int, n_classes: int, invtemp: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
def calib_dataset(request) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_classes, invtemp, random_state = request.param
    rng = np.random.default_rng(random_state)
    alpha = rng.exponential(scale=1, size=n_classes)
    probs = rng.dirichlet(alpha, size=n_samples)
    true_probs = scipy.special.softmax(invtemp * np.log(probs + 1e-30), axis=1)
    y = sample_labels(true_probs)
    probs = probs.astype(np.float32)  # todo
    return probs, y


# possible tests:
# test that they do something sensible (reduce Brier loss on a very bad predictor)
# test that they work with multiclass?
# test that torch and numpy interfaces are equivalent?

# @pytest.mark.parametrize('calibrators', [
#     (VennAbersCalibrator(use_ovo=False), MulticlassOneVsRestCalibrator(BinaryVennAbersCalibrator())),
#     (VennAbersCalibrator(use_ovo=True), MulticlassOneVsOneCalibrator(BinaryVennAbersCalibrator())),
#     (SklearnCalibrator(method='isotonic', cv='prefit'),
#      MulticlassOneVsRestCalibrator(SklearnCalibrator(method='isotonic', cv='prefit')))
# ])
# def test_calibrators_equal(calibrators: Tuple[Calibrator, Calibrator], calib_dataset: Tuple[np.ndarray, np.ndarray]):
#     cal1, cal2 = calibrators
#     cal1 = sklearn.base.clone(cal1)
#     cal2 = sklearn.base.clone(cal2)
#
#     X, y = calib_dataset
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#
#     cal1.fit(X, y)
#     cal2.fit(X, y)
#
#     # assert np.allclose(cal1.predict_proba(X_test), cal2.predict_proba(X_test))
#     np.testing.assert_allclose(cal1.predict_proba(X_test), cal2.predict_proba(X_test), atol=1e-7)

@pytest.mark.parametrize('metric', [BrierLoss(), SmoothCalibrationError(), CalibrationError()])
@pytest.mark.parametrize('calibrator', [
    get_calibrator(name) for name in
    ['platt', 'isotonic', 'platt-logits', 'ivap-ovr', 'ivap-ovo', 'cir', 'temp-scaling', 'autogluon-ts',
     # 'dircal', 'dircal-cv',
     'torchunc-ts', ]  # don't test guo because it's too bad
] + [TemperatureScalingCalibrator(opt='lbfgs'), get_calibrator('temp-scaling', calibrate_with_mixture=True)])
def test_calibrator_performance(metric: ClassificationMetric, calibrator: Calibrator,
                                calib_dataset: Tuple[np.ndarray, np.ndarray]):
    X, y = calib_dataset
    # don't test with train/test split since it might fail due to overfitting
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    #
    # cal = sklearn.base.clone(calibrator)
    # cal.fit(X_train, y_train)
    # y_pred_probs = cal.predict_proba(X_test)
    #
    # n_classes = X.shape[-1]
    # y_true = CategoricalDirac(torch.as_tensor(y_test), n_classes)
    # without_cal = metric.compute(y_true=y_true, y_pred=CategoricalProbs(torch.as_tensor(X_test))).item()
    # with_cal = metric.compute(y_true=y_true, y_pred=CategoricalProbs(torch.as_tensor(y_pred_probs))).item()

    cal = sklearn.base.clone(calibrator)
    cal.fit(X, y)
    y_pred_probs = cal.predict_proba(X)

    n_classes = X.shape[-1]
    y_true = CategoricalDirac(torch.as_tensor(y), n_classes)
    without_cal = metric.compute(y_true=y_true, y_pred=CategoricalProbs(torch.as_tensor(X))).item()
    with_cal = metric.compute(y_true=y_true, y_pred=CategoricalProbs(torch.as_tensor(y_pred_probs))).item()

    # loss after calibration should be better than before
    assert with_cal < without_cal


@pytest.mark.parametrize('calibrator', [
    get_calibrator(name) for name in
    ['platt', 'isotonic', 'platt-logits', 'ivap-ovr', 'ivap-ovo', 'cir', 'temp-scaling',
     # 'dircal', 'dircal-cv',
     'ivap', 'autogluon-ts', 'guo-ts', 'autogluon-ts', ]
] + [TemperatureScalingCalibrator(opt='lbfgs'), get_calibrator('temp-scaling', calibrate_with_mixture=True)])
def test_calibrator_torch_vs_numpy(calibrator: Calibrator, calib_dataset: Tuple[np.ndarray, np.ndarray]):
    X, y = calib_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    n_classes = X.shape[-1]

    preds = []
    for train_torch in [False, True]:
        cal = sklearn.base.clone(calibrator)
        if train_torch:
            cal.fit_torch(CategoricalProbs(torch.as_tensor(X_train)), torch.as_tensor(y_train))
        else:
            cal.fit(X_train, y_train)

        for predict_torch in [False, True]:
            if predict_torch:
                y_pred = cal.predict_proba_torch(CategoricalProbs(torch.as_tensor(X_test))).get_probs().numpy()
            else:
                y_pred = cal.predict_proba(X_test)
            preds.append(y_pred)

    for i in range(1, 4):
        np.testing.assert_allclose(preds[i], preds[0], atol=1e-7)


@pytest.mark.parametrize('calibrator', [
    get_calibrator(name) for name in
    ['platt', 'isotonic', 'platt-logits', 'ivap-ovr', 'ivap-ovo',
     # 'dircal', 'dircal-cv',
     'cir', 'temp-scaling', 'autogluon-ts', 'guo-ts', 'torchunc-ts']
] + [TemperatureScalingCalibrator(opt='lbfgs'), get_calibrator('temp-scaling', calibrate_with_mixture=True)])
def test_calibrator_missing_class(calibrator: Calibrator):
    rng = np.random.default_rng(0)
    n_samples = 1000
    X = rng.normal(size=(n_samples, 4))
    y = rng.choice([0, 2, 3], size=n_samples, replace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    cal = sklearn.base.clone(calibrator)
    cal.fit(X_train, y_train)
    y_pred_probs = cal.predict_proba(X_test)

    assert X.shape[-1] == y_pred_probs.shape[-1]
