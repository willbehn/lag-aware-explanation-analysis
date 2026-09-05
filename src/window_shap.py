import numpy as np
import shap


def split_data_in_windows(width: int, max_width: int) -> list[list[int]]:
    return [list(range(i, min(i + width, max_width))) for i in range(0, max_width, width)]

# def shap_per_lag

# def shap_truth

def explain(model, window_width, series):
    windows = split_data_in_windows(width=window_width, max_width=len(series))
