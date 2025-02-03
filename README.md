# Probmetrics: Classification metrics and post-hoc calibration

This package (PyTorch-based) currently contains
- classification metrics, especially also 
metrics for assessing the quality of probabilistic predictions, and
- post-hoc calibration methods, especially 
  a fast and accurate implementation of temperature scaling.

It accompanies our paper
[Rethinking Early Stopping: Refine, Then Calibrate](https://arxiv.org/abs/2501.19195).
Please cite our paper if you use this repository for research purposes.
The experiments from the paper can be found here: 
[vision](https://github.com/eugeneberta/RefineThenCalibrate-Vision), 
[tabular](https://github.com/dholzmueller/pytabkit), 
[theory](https://github.com/eugeneberta/RefineThenCalibrate-Theory).

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

## Using temperature scaling

We provide a highly efficient implementation of temperature scaling
that, unlike some other implementations, 
does not suffer from optimization issues.

### Numpy interface

```python
from probmetrics.calibrators import get_calibrator
import numpy as np

probas = np.asarray([[0.1, 0.9]])
labels = np.asarray([1])
# this is the version with Laplace smoothing, 
# use 'temp-scaling' for the version without
calib = get_calibrator('temp-scaling', calibrate_with_mixture=True)
# other option: calib = MixtureCalibrator(TemperatureScalingCalibrator())
# there is also a fit_torch / predict_proba_torch interface
calib.fit(probas, labels)
calibrated_probas = calib.predict_proba(probas)

# -------- alternatively, using torch tensors (GPU support) ------------

```

### PyTorch interface

The PyTorch version can be used directly with GPU tensors, but 
this can actually be slower than CPU for smaller validation sets (around 1K-10K samples).

```python
from probmetrics.distributions import CategoricalProbs
from probmetrics.calibrators import get_calibrator
import torch

probas = torch.as_tensor([[0.1, 0.9]])
labels = torch.as_tensor([1])

calib = get_calibrator('ts-mix')

# if you have logits, you can use CategoricalLogits instead
calib.fit_torch(CategoricalProbs(probas), labels)
calib.predict_proba_torch(CategoricalProbs(probas))
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