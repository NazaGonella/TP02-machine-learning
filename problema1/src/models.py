import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, x : np.ndarray, b : np.ndarray, L1 : float = 0, L2 : float = 0, initial_weight_value : float = 1):
        self.x : np.ndarray = np.array(np.c_[np.ones(x.shape[0]), x], dtype=np.float64)   # agrego columna de unos para el bias.
        self.w : np.ndarray = np.full(shape=self.x.shape[1], fill_value=initial_weight_value)
        self.b : np.ndarray = np.array(b, dtype=np.float64)
        self.L1 = L1
        self.L2 = L2
    
    def fit_logistic_regression(self, step_size : float, tolerance : float = -1, max_number_of_steps : int = -1):
        attempts = 0
        while True:
            gradient = self.least_squares_gradient()
            if (np.linalg.norm(gradient) <= tolerance and tolerance != -1) or (attempts >= max_number_of_steps and max_number_of_steps != -1):
                break
            self.w = self.w - (step_size * (gradient))
            attempts += 1

    def logistic_sigmoid_function(x : float) -> float:
        return np.log(x/(1-x))

    def f_score(tp : int, fp : int, fn : int) -> float:
        return (2*tp) / ((2*tp) + fp + fn)

    def predict(self, input : np.ndarray) -> np.ndarray:
        return self.w @ input

    def print_weights(self, weight_names : list[str]) -> None:
        print(f'{'BIAS':14}', '(w0): ', self.w[0])
        for i in range(self.x.shape[1] - 1):
            print(f'{weight_names[i]:14} (w{i+1}): ', self.w[i+1])
