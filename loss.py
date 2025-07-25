import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def d_mse(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
