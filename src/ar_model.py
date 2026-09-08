"""
Simple auto regressive model that have declared specific lags that are used, to 
predict the next t-1 time step

x_t = delta + cof*x_(t-1) + alpha

"""
import numpy as np
from abc import ABC, abstractmethod

class ARModel(ABC):
    @abstractmethod
    def predict(self, series) -> float:
        ...

class SingleFeatureARModel(ARModel):
    def __init__(self, used_lags: dict[int, float], X):
        self._used_lags = used_lags 
        self._mean_avg = np.mean(X)
        self._delta = (1 - sum(self._used_lags.values())) * self._mean_avg

        # declared lags = 

    def predict(self, series) -> float:
        return self._delta + sum(w * series[k - 1] for k, w in self._used_lags.items())


class MultiFeatureARModel(ARModel):
    def __init__(self, features, used_lags: dict[str, dict[int, float]], X):
        self._features = features
        self._used_lags = used_lags
        self._deltas = self.calculate_delta(X)

    def calculate_delta(self, X) -> dict[str, float]:
        out = {}
        for feat in self._features:
            out[feat] = (1 - sum(self._used_lags[feat].values())) * np.mean(X[feat])

        return out

    def predict(self, series) -> dict[str, float]:
        out = {}

        for feat in self._features:
            out[feat] = self._deltas[feat] + sum(w * series[feat][k - 1] for k, w in self._used_lags[feat].items())

        return out