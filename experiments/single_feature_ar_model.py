from ar_model import SingleFeatureARModel
import numpy as np

def main():
    used_lags = {2: 0.25, 4: 0.65}
    length = 12

    rng = np.random.default_rng(123)
    x = rng.normal(50, 12, length)

    ar_1 = SingleFeatureARModel(used_lags=used_lags)
    print(ar_1.predict(x))


if __name__ == "__main__":
    main()
