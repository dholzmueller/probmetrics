import numpy as np
import sklearn
import pandas as pd
import torch

from probmetrics.calibrators import get_calibrator
from probmetrics.utils import multiclass_probs_to_logits


class WS_CatboostClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, iterations=10, use_init_logits=True, early_stopping_rounds=300, thread_count=1, random_state=0, verbose=0):
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.thread_count = thread_count
        self.random_state = random_state
        self.use_init_logits = use_init_logits
        self.verbose = verbose

    def _fit_model(self, idxs):
        from catboost import CatBoostClassifier, Pool
        train_idx, val_idx = idxs[0], idxs[1]
        X_train, y_train = self.X_.take(train_idx, 0), self.y_[train_idx]
        X_val, y_val = self.X_.take(val_idx, 0), self.y_[val_idx]

        train_pool = Pool(X_train, label=y_train)
        val_pool = Pool(X_val, label=y_val)

        if self.init_logits_ is not None:
            train_pool.set_baseline(self.init_logits_.take(train_idx, 0))
            val_pool.set_baseline(self.init_logits_.take(val_idx, 0))

        m = CatBoostClassifier(
            iterations=self.iterations,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds,
            thread_count=self.thread_count,
            verbose=self.verbose,
            loss_function='MultiClass',
            classes_count=len(self.classes_),
        )

        return  m.fit(train_pool, eval_set=val_pool)

    def _get_model_proba(self, model, X):
        """Calculates probabilities by adding baseline to raw residuals."""
        from scipy.special import expit, softmax
        raw_preds = model.predict(X, prediction_type='RawFormulaVal')
        init_score = multiclass_probs_to_logits(X) if self.use_init_logits else None
        if init_score is not None:
            raw_preds = raw_preds + init_score
        
        return softmax(raw_preds, axis=1)

    def fit(self, X, y):
        import multiprocessing as mp
        if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)): y = y.values
        
        init_logits = multiclass_probs_to_logits(X) if self.use_init_logits else None

        self.init_logits_ = init_logits
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        
        splits = list(sklearn.model_selection.StratifiedKFold(
            n_splits=8, shuffle=True, random_state=self.random_state
        ).split(X, y))
        
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        
        oof_preds_list = []
        for m, idxs in zip(self.models_, splits):
            val_idx = idxs[1]
            oof_preds_list.append(self._get_model_proba(m, X.take(val_idx, 0)))

        oof_preds = np.concatenate(oof_preds_list, axis=0)
        val_indices = np.concatenate([idxs[1] for idxs in splits])
        oof_labels = y[val_indices]
        
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
        avg_probas = np.mean([self._get_model_proba(m, X) for m in self.models_], axis=0)
        return self.calib_.predict_proba(avg_probas)

    def predict(self, X):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))


class WS_LGBMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, n_estimators=10, use_init_logits=True, learning_rate=0.04, subsample=0.75, num_leaves=50, subsample_freq=1, random_state=0, early_stopping_round=100, min_child_samples=40, min_child_weight=1e-7, n_jobs=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.num_leaves = num_leaves
        self.subsample_freq = subsample_freq
        self.random_state = random_state
        self.early_stopping_round = early_stopping_round
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.n_jobs = n_jobs
        self.use_init_logits = use_init_logits


    def _fit_model(self, idxs):
        from lightgbm import LGBMClassifier
        train_idx, val_idx = idxs[0], idxs[1]
        X_train = self.X_.take(train_idx, 0)
        y_train = self.y_[train_idx]
        X_val = self.X_.take(val_idx, 0)
        y_val = self.y_[val_idx]

        fit_params = {}
        if self.init_logits_ is not None:
            fit_params['init_score'] = self.init_logits_.take(train_idx, 0)
            fit_params['eval_init_score'] = [self.init_logits_.take(val_idx, 0)]

        m = LGBMClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate, subsample=self.subsample, subsample_freq=self.subsample_freq, num_leaves=self.num_leaves,
                           random_state=self.random_state, early_stopping_round=self.early_stopping_round, min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                           n_jobs=self.n_jobs, verbosity=self.verbose_, objective="multiclass", num_class=len(self.classes_))
        
        return m.fit(X_train, y_train, eval_set=(X_val, y_val), **fit_params)

    def _get_model_proba(self, model, X):
        from scipy.special import softmax
        init_score = multiclass_probs_to_logits(X) if self.use_init_logits else None

        raw_preds = model.predict(X, raw_score=True)
        if init_score is not None: raw_preds += init_score 

        return softmax(raw_preds, axis=1)

    def fit(self, X, y, verbose=-1):
        import multiprocessing as mp
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        
        init_logits = multiclass_probs_to_logits(X) if self.use_init_logits else None

        self.verbose_ = verbose
        self.init_logits_ = init_logits
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        
        oof_preds_list = []
        for m, idxs in zip(self.models_, splits):
            val_idx = idxs[1]
            oof_preds_list.append(self._get_model_proba(m, X.take(val_idx, 0)))

        oof_preds = np.concatenate(oof_preds_list, axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)

        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        
        avg_probas = np.mean([self._get_model_proba(m, X) for m in self.models_], axis=0)
        
        return self.calib_.predict_proba(avg_probas)

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

