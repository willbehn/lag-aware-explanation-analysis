import numpy as np
import shap
from itertools import product
from sklearn.metrics import mean_squared_error

from ar_model import SingleFeatureARModel
from ar_model import MultiFeatureARModel

def split_data_in_windows(width: int, max_width: int) -> list[list[int]]:
    return [list(range(i, min(i + width, max_width))) for i in range(0, max_width, width)]

# SHAP paper 4.2, linear shap, true shap values can be calculated excact for linear models
def shap_truth(used_lags: dict[int, float], X) -> np.ndarray:
    truth = np.zeros(len(X))
    mean_avg = np.mean(X)

    for k, w in used_lags.items():
        truth[k - 1] = w * (X[k - 1] - mean_avg)

    return truth

# WindowSHAP paper formula 5, shap values can be projected to individual lags by dividing on the window size
def shap_per_lag(windows, num_lags, shap_values) -> np.ndarray:
    lags = np.zeros(num_lags)

    for window, sv in zip(windows, shap_values, strict=True):
        lags[window] = sv/len(window)

    return np.array(lags)

# MSE between true shap values per lag and the shap values per lag 
# after explaining based on window size. TODO not normalized
def attribution_mse(windows, shap_values, truth: np.ndarray) -> float:
    shap_lags = shap_per_lag(windows=windows, num_lags=len(truth), shap_values=shap_values)

    return mean_squared_error(y_true=truth, y_pred=shap_lags)

def explain(
        model: SingleFeatureARModel,
        window_width: int, 
        X: np.ndarray
        ) -> tuple[list[list[int]], np.ndarray]:
    
    windows = split_data_in_windows(width=window_width, max_width=len(X))

    num_windows = len(windows)
    baseline = float(X.mean())

    def value_function(mask) -> np.ndarray:
        output = []
        mask = np.atleast_2d(mask)

        for m in mask:
            v = X.copy()

            # feks (1, [0,1]) if window 0 is on, else (0, [0,1]) if its off
            for on, win in zip(m, windows):
                if not on: 
                    v[win] = baseline #TODO look into replacing with real data, not mean

            output.append(model.predict(v))

        return np.array(output)

    explainer = shap.KernelExplainer(value_function, np.zeros((1, num_windows)))
    shap_values = explainer.shap_values(np.ones(num_windows))

    return windows, np.array(shap_values).ravel()


def explain_multi(
        model: MultiFeatureARModel, 
        window_width, X_multi: dict[str, np.ndarray], 
        target: str
        ) -> tuple[list[list[int]], np.ndarray]:
   
    # assumes that all features will have the same max width
    windows = split_data_in_windows(width=window_width, max_width=len(next(iter(X_multi.values()))))
    num_players = len(windows)*len(X_multi)

    baseline = {}

    features = list(X_multi.keys())

    for feat in features:
        baseline[feat] = np.array(X_multi[feat]).mean()

    def value_function(mask_matrix) -> np.ndarray:
        # mask_matrix shape = (n, num_windows)
        output = []
        mask = np.atleast_2d(mask_matrix)

        for m in mask:
            v = {feat: X.copy() for feat, X in X_multi.items()}

            for on, (feat, window) in zip(m, product(features, windows)):
                if not on: 
                    v[feat][window] = baseline[feat]

            output.append(model.predict(v)[target])

        return np.array(output)

    explainer = shap.KernelExplainer(value_function, np.zeros((1, num_players)))
    shap_values = explainer.shap_values(np.ones(num_players))

    # Slices the shap_values to only contain the windows for target feature
    target_start = features.index(target)*len(windows)
    target_end = target_start + len(windows)
    shap_values_target = shap_values[target_start : target_end] 

    return windows, np.array(shap_values_target).ravel()