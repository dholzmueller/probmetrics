[![test](https://github.com/dholzmueller/probmetrics/actions/workflows/testing.yml/badge.svg)](https://github.com/dholzmueller/probmetrics/actions/workflows/testing.yml)
[![Downloads](https://img.shields.io/pypi/dm/probmetrics)](https://pypistats.org/packages/probmetrics)


# Probmetrics: Classification metrics and post-hoc calibration

This package (PyTorch-based) currently contains
- classification metrics, especially also 
metrics for assessing the quality of probabilistic predictions, and
- post-hoc calibration methods, especially
  - a fast and accurate implementation of temperature scaling.
  - an implementation of structured matrix scaling (SMS), 
    a regularized version of matrix scaling that outperforms other 
    logistic-based calibration functions.

It accompanies our papers
[Rethinking Early Stopping: Refine, Then Calibrate](https://arxiv.org/abs/2501.19195) and [Structured Matrix Scaling for Multi-Class Calibration](https://arxiv.org/abs/2511.03685).
Please cite us if you use this repository for research purposes.
The experiments from the papers can be found here: 
- Rethinking Early Stopping:
  - [vision experiments](https://github.com/eugeneberta/RefineThenCalibrate-Vision).
  - [tabular experiments](https://github.com/dholzmueller/pytabkit).
  - [theory](https://github.com/eugeneberta/RefineThenCalibrate-Theory).
- Structured Matrix Scaling: 
  [all experiments](https://github.com/eugeneberta/LogisticCalibrationBenchmark).

## Installation

Probmetrics is available via
```bash
pip install probmetrics
```
To obtain all functionality, install `probmetrics[extra,dev,dirichletcal]`.
- extra installs more packages for smooth ECE, 
  Venn-Abers calibration, 
  centered isotonic regression, 
  the temperature scaling implementation in NetCal.
- dev installs more packages for development (esp. documentation)
- dirichletcal installs Dirichlet calibration, 
  which however only works for Python 3.12 upwards.

## Using post-hoc calibration methods

You can create a calibrator as follows:
```python
from probmetrics.calibrators import get_calibrator

calib = get_calibrator('logistic')
```

These are the main supported methods:
- `'logistic'` defaults to structured matrix scaling (SMS) for multiclass 
  and quadratic scaling for binary calibration. 
  We recommend using `'logistic'` for best results, 
  especially on multiclass problems. 
  It can be slow for larger numbers of classes. Only runs on CPU. 
  For the (L-)BFGS version (not the default), 
  the first call is slower due to numba compilation.
- `'svs'`: Structured vector scaling (SVS) for multiclass problems, 
  faster than SMS for multiclass while being almost as good in many cases.
- `'affine-scaling'`: Affine scaling for binary problems, 
  underperforms `'logistic'` (quadratic scaling) in our benchmarks but preserves AUC.
- `'temp-scaling'`: Our 
  [highly efficient implementation of temperature scaling](https://arxiv.org/abs/2501.19195)
  that, unlike some other implementations, 
  does not suffer from optimization issues. 
  Temperature scaling is not as expressive as matrix or vector scaling variants,
  but it is faster and has the least overfitting risk.
- `'ts-mix'`: Same as `'temp-scaling'` but with Laplace smoothing 
  (slightly preferable for logloss). Can also be achieved using 
  `get_calibrator('temp-scaling', calibrate_with_mixture=True)`
- `'isotonic'` Isotonic regression from scikit-learn. 
  Isotonic variants can be good for binary classification with enough data (around 10K samples or more)
- `'ivap'` Inductive Venn-Abers predictor (a version of isotonic regression, slow but a bit better)
- `'cir'` Centered isotonic regression (slightly better and slower than isotonic)
- `'dircal'` Dirichlet calibration (slow, logistic performs better in our experiments)
- `'dircal-cv'` Dirichlet calibration optimized with cross-validation (very slow)

More details on parameters and other methods can be found in the get_calibrator function 
[here](https://github.com/dholzmueller/probmetrics/probmetrics/calibrators.py).

### Usage with `numpy`

```python
import numpy as np

probas = np.asarray([[0.1, 0.9]])  # shape = (n_samples, n_classes)
labels = np.asarray([1])  # shape = (n_samples,)
calib.fit(probas, labels)
calibrated_probas = calib.predict_proba(probas)
```

### Usage with PyTorch

The PyTorch version can be used directly with GPU tensors, 
which is leveraged by our temperature scaling implementation 
but not by most other methods.
For temperature scaling, this could accelerate things, 
but the CPU version can be faster 
for smaller validation sets (around 1K-10K samples).

```python
from probmetrics.distributions import CategoricalProbs
import torch

probas = torch.as_tensor([[0.1, 0.9]])
labels = torch.as_tensor([1])

# if you have logits, you can use CategoricalLogits instead
calib.fit_torch(CategoricalProbs(probas), labels)
result = calib.predict_proba_torch(CategoricalProbs(probas))
calibrated_probas = result.get_probs()
```


## Using our refinement and calibration metrics

We provide estimators for refinement error 
(loss after post-hoc calibration)
and calibration error 
(loss improvement through post-hoc calibration). 
They can be used as follows:

```python
import torch
from probmetrics.metrics import Metrics

# compute multiple metrics at once 
# this is more efficient than computing them individually
metrics = Metrics.from_names(['logloss', 
                              'refinement_logloss_ts-mix_all', 
                              'calib-err_logloss_ts-mix_all'])
y_true = torch.tensor(...)
y_logits = torch.tensor(...)
results = metrics.compute_all_from_labels_logits(y_true, y_logits)
print(results['refinement_logloss_ts-mix_all'].item())
```

## Using more metrics

In general, while some metrics can be 
flexibly configured using the corresponding classes,
many metrics are available through their name. 
Here are some relevant classification metrics:
```python
from probmetrics.metrics import Metrics

metrics = Metrics.from_names([
    'logloss',
    'brier',  # for binary, this is 2x the brier from sklearn
    'accuracy', 'class-error',
    'auroc-ovr', # one-vs-rest
    'auroc-ovo-sklearn', # one-vs-one (can be slow!)
    # calibration metrics
    'ece-15', 'rmsce-15', 'mce-15', 'smece'
    'refinement_logloss_ts-mix_all', 
    'calib-err_logloss_ts-mix_all',
    'refinement_brier_ts-mix_all', 
    'calib-err_brier_ts-mix_all'
])
```

The following function returns a list of all metric names:
```python
from probmetrics.metrics import Metrics, MetricType
Metrics.get_available_names(metric_type=MetricType.CLASS)
```

While there are some classes for regression metrics, they are not implemented.

## Contributors
- David Holzmüller
- Eugène Berta

## Releases

- v1.1.0 by @eugeneberta: Improvements to the SVS and SMS calibrators:
  - logit pre-processing with `'ts-mix'` is now automatic, 
    and the global scaling parameter $\alpha$ is fixed to 1. This yields:
    - improved performance on our tabular and computer vision benchmarks 
      (see the arxiv v2 of the SMS paper, coming soon).
    - faster convergence.
    - ability to compute the duality gap in closed form for stopping SAGA solvers, 
      which we implement in this version.
  - improved L-BFGS solvers, much faster than in the previous version. 
    Now the solver for default SVS and SMS.
  - the default binary calibrator in `LogisticCalibrator` is now quadratic scaling 
    instead of affine scaling, this can be changed back by using 
    `LogisticCalibrator(binary_type='affine')`.
- v1.0.0 by @eugeneberta: New post-hoc calibrators like `'logistic'` 
  including structured matrix scaling (SMS), 
  structured vector scaling (SVS), 
  affine scaling, and quadratic scaling.
- v0.0.2 by @dholzmueller:
  - Removed numpy<2.0 constraint
  - allow 1D vectors in CategoricalLogits / CategoricalProbs
  - add TorchCal temperature scaling
  - minor fixes in AutoGluon temperature scaling 
    that shouldn't affect the performance in practice
- v0.0.1 by @dholzmueller: Initial release