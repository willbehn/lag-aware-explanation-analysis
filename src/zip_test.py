from itertools import product

m = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
features = ["feature_1", "feature_2"]
windows = [[1, 5], [2, 10]]

result = list(zip(m, product(features, windows)))

for element in result:
    print(element)