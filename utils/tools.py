import numpy as np


def transition_matrix_error(T, T_hat):
    return np.sum(np.abs(T-T_hat)) / np.sum(np.abs(T))