import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import preprocessing as prepro
# import data_handler
# from utils import RawData

class LinealReg:
    def __init__(self, x : np.ndarray, y : np.ndarray, L1 : float = 0, L2 : float = 0, initial_weight_value : float = 1):
        self.x : np.ndarray = np.array(np.c_[np.ones(x.shape[0]), x], dtype=np.float64)   # agrego columna de unos para el bias.
        self.y : np.ndarray = np.array(y, dtype=np.float64)
        self.L1 = L1
        self.L2 = L2
        self.coef : np.ndarray = np.full(shape=self.x.shape[1], fill_value=initial_weight_value)
    
    def fit_pseudo_inverse(self):
        self.coef = np.linalg.inv((self.x.T @ self.x) + (self.L2 * np.identity(self.x.shape[1]))) @ self.x.T @ self.y

    def fit_gradient_descent(self, step_size : float, tolerance : float = -1, max_number_of_steps : int = -1):
        attempts = 0
        while True:
            gradient = self.least_squares_gradient()
            if (np.linalg.norm(gradient) <= tolerance and tolerance != -1) or (attempts >= max_number_of_steps and max_number_of_steps != -1):
                break
            self.coef = self.coef - (step_size * (gradient))
            attempts += 1

    def error_least_squares_function(self) -> float:
        # ||Xw - Y||^2
        return np.linalg.norm((self.x @ self.coef) - self.y)**2

    def error_cuadratico_medio(self, validation_set_y : np.ndarray = None, validation_set_x : np.ndarray = None) -> float:
        val_set_y : np.ndarray = self.y
        val_set_x : np.ndarray = self.x
        if validation_set_y is not None and validation_set_x is not None:
            val_set_y = validation_set_y
            val_set_x = np.array(np.c_[np.ones(validation_set_x.shape[0]), validation_set_x], dtype=np.float64)
        sum : float = 0
        result = (val_set_y - (val_set_x @ self.coef))**2
        for i in range(val_set_y.shape[0]):
            sum += result[i]
        return sum / val_set_y.shape[0]
    
    def least_squares_gradient(self) -> np.ndarray:
        # 2X^T * (Xw - Y)
        return ((2 * self.x.T) @ ((self.x @ self.coef) - self.y)) + (2 * self.L2 * self.coef) + (self.L1 * np.sign(self.coef))

    def predict(self, input : np.ndarray) -> np.ndarray:
        return self.coef @ input

    def print_coef(self, weight_names : list[str]) -> None:
        print(f'{'BIAS':14}', '(w0): ', self.coef[0])
        for i in range(self.x.shape[1] - 1):
            print(f'{weight_names[i]:14} (w{i+1}): ', self.coef[i+1])
