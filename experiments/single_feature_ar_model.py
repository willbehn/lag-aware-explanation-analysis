from ar_model import SingleFeatureARModel
from window_shap import explain, shap_per_lag
import numpy as np
import shap

def main():
    used_lags = {2: 0.25, 4: 0.65}
    length = 12

    rng = np.random.default_rng(123)
    X = rng.normal(50, 12, length)

    ar_1 = SingleFeatureARModel(used_lags=used_lags, X=X)

    window_sizes = {1,2,4,6,8,10,12}
    labels = [f"window {i}" for i in range(len(window_sizes))]

    for win in window_sizes:
        print(f"window size {win}:")
        windows, shap_values = explain(ar_1,win,X)
        shap_pl = shap_per_lag(windows, len(X), shap_values)
        print(shap_values)

        print(f"SHAP value per lag:\n {shap_pl}")

        #explanation = shap.Explanation(shap_values, feature_names=labels)
        #shap.plots.bar(explanation)
  
if __name__ == "__main__":
    main()
