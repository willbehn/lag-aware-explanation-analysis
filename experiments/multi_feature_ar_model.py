from ar_model import MultiFeatureARModel
from window_shap import explain, shap_per_lag
import numpy as np
import shap

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

if __name__ == "__main__":
    main()
