from typing import List, Tuple, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

from probmetrics.distributions import Distribution, CategoricalDistribution


class Splitter:
    def get_splits(self, y_true: Distribution, random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    def get_name(self) -> str:
        raise NotImplementedError()


class AllSplitter(Splitter):
    def get_splits(self, y_true: Distribution, random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(np.arange(y_true.get_n_samples()), np.arange(y_true.get_n_samples()))]

    def get_name(self) -> str:
        return 'all'


class CVSplitter(Splitter):
    def __init__(self, n_cv: int):
        self.n_cv = n_cv

    def get_splits(self, y_true: Distribution, random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if isinstance(y_true, CategoricalDistribution):
            skf = StratifiedKFold(n_splits=self.n_cv, shuffle=True, random_state=random_state)
            return skf.split(X=np.zeros(y_true.get_n_samples()), y=y_true.get_modes().numpy())
        else:
            kf = KFold(n_splits=self.n_cv, random_state=random_state)
            return kf.split(X=np.zeros(y_true.get_n_samples()))

    def get_name(self) -> str:
        return f'cv-{self.n_cv}'
