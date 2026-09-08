import numpy as np
import shap

from ar_model import ARModel

def split_data_in_windows(width: int, max_width: int) -> list[list[int]]:
    return [list(range(i, min(i + width, max_width))) for i in range(0, max_width, width)]

# def shap_per_lag

# SHAP paper 4.2, linear shap, true shap values can be calculated excact for linear models
def shap_truth(used_lags: dict[int, float], series):
    truth = np.zeros(len(series))
    mean_avg = np.mean(series)

    for k, w in used_lags.items():
        truth[k - 1] = w * (series[k - 1] - mean_avg)

    return truth

# WindowSHAP paper formula 5, shap values can be projected to individual lags by dividing on the window size
def shap_per_lag(windows, num_lags, shap_values) -> np.ndarray:
    shap_per_lag = np.zeros(num_lags)

    for window, sv in zip(windows, shap_values):
        shap_per_lag[window] = sv/len(window)

    return np.array(shap_per_lag)


def explain(model: ARModel, window_width, X) -> tuple[list[list[int]], np.ndarray]:
    windows = split_data_in_windows(width=window_width, max_width=len(X))

    num_windows = len(windows)
    baseline = float(X.mean())

    def value_function(mask) -> np.ndarray:
        output = []
        mask = np.atleast_2d(mask)

        for m in mask:
            v = X.copy()

            # feks (1, [0,1]) if window 0 is on, else (0, [0,1]) if its off
            for on, win in zip (m, windows):
                if not on: 
                    v[win] = baseline #TODO look into replacing with real data, not mean

            output.append(model.predict(v))

        return np.array(output)

    explainer = shap.KernelExplainer(value_function, np.zeros((1, num_windows)))
    shap_values = explainer.shap_values(np.ones(num_windows))

    return windows, np.array(shap_values).ravel()


