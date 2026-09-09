from ar_model import MultiFeatureARModel
from window_shap import explain_multi, shap_truth, attribution_mse
import numpy as np

def main():
    features = ["feature_1", "feature_2"]
    used_lags = {
        "feature_1": {2: 0.25, 4: 0.65},
        "feature_2": {1: 0.50, 3: 0.20},
    }
    length = 12

    rng = np.random.default_rng(123)
    X = {
        "feature_1": rng.normal(50, 12, length),
        "feature_2": rng.normal(20, 5, length),
    }

    ar_model = MultiFeatureARModel(features=features, used_lags=used_lags, X=X)
    print(f"\nprediction: {ar_model.predict(X)}")

    target = "feature_2"

    windows, shap_values = explain_multi(ar_model, 12, X, target=target)
    truth = shap_truth(used_lags=used_lags[target], X=X[target])

    print(f"windows: {windows}")
    print(f"shap values: {shap_values}")
    print(f"shap truth: {truth}")
    print(f"mse: {attribution_mse(windows=windows, shap_values=shap_values, truth=truth)}")

    #print(explain_multi(ar_model, 1, X, target=target))
    #print(shap_truth(used_lags=used_lags[target], X=X[target]))

if __name__ == "__main__":
    main()
