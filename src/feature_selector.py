import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import roc_auc_score


class SequentialFeatureSelectorWrapper:
    
    def __init__(
        self,
        model,
        n_features_to_select=10,
        direction="forward",
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    ):
        self.model = model
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        
        self.selected_features_ = None
        self.selector_ = None
        self.cv_score_ = None

    def fit(self, X, y):
        
        model = clone(self.model)

        cv_strategy = TimeSeriesSplit(n_splits=self.cv)

        self.selector_ = SequentialFeatureSelector(
            model,
            n_features_to_select=self.n_features_to_select,
            direction=self.direction,
            scoring=self.scoring,
            cv=cv_strategy,
            n_jobs=self.n_jobs
        )

        self.selector_.fit(X, y)

        self.selected_features_ = X.columns[self.selector_.get_support()].tolist()

        # Calcular score medio después de selección
        scores = self.evaluate(X[self.selected_features_], y)
        self.cv_score_ = np.mean(scores)

        return self

    def transform(self, X):
        return X[self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self):
        return self.selected_features_

    def get_cv_score(self):
        return self.cv_score_

    def evaluate(self, X, y):
        
        tscv = TimeSeriesSplit(n_splits=self.cv)
        scores = []

        for train_idx, test_idx in tscv.split(X):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = clone(self.model)
            model.fit(X_train, y_train)

            y_pred = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred)

            scores.append(score)

        return scores
