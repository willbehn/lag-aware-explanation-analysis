from ar_model import SingleFeatureARModel
from window_shap import explain
import numpy as np

def main():
    used_lags = {2: 0.25, 4: 0.65}
    length = 12

    rng = np.random.default_rng(123)
    X = rng.normal(50, 12, length)

    ar_1 = SingleFeatureARModel(used_lags=used_lags)

    print(explain(ar_1,2,X))
  


if __name__ == "__main__":
    main()
