"""
Simple auto regressive model that have declared specific lags that are used, to 
predict the next t-1 time step

x_t = delta + cof*x_(t-1) + alpha

"""
import numpy as np

class SingleFeatureARModel:
    def __init__(self, used_lags: dict[int, float], X):
        self._used_lags = used_lags 
        self._mean_avg = np.mean(X)
        self._delta = (1 - sum(self._used_lags.values())) * self._mean_avg

        # declared lags = 

    def predict(self, series) -> float:
        return self._delta + sum(w * series[k - 1] for k, w in self._used_lags.items())
