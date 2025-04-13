import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

class LogisticRegression:
    def __init__(self, x : np.ndarray, b : np.ndarray, L2 : float = 0, initial_weight_value : float = 1):
        self.x : np.ndarray = np.array(np.c_[np.ones(x.shape[0]), x], dtype=np.float64)   # agrego columna de unos para el bias.
        self.w : np.ndarray = np.full(shape=self.x.shape[1], fill_value=initial_weight_value)
        self.b : np.ndarray = np.array(b, dtype=np.float64)
        self.L2 : int = L2
        self.tp : int = 0
        self.tn : int = 0
        self.fp : int = 0
        self.fn : int = 0
        self.pred_probs : np.ndarray[float] = np.array([])

    def fit_gradient_descent(self, step_size : float, tolerance : float = -1, max_number_of_steps : int = -1):
        attempts = 0
        while True:
            gradient = self.gradiente_cross_entropy()
            if (np.linalg.norm(gradient) <= tolerance and tolerance != -1) or (attempts >= max_number_of_steps and max_number_of_steps != -1):
                break
            self.w = self.w - (step_size * (gradient))
            attempts += 1

    def sigmoid_function(self, x : float) -> float:
        return 1/(1+np.exp(-x))
    
    def binary_cross_entropy(self) -> float:
        pred : float = self.sigmoid_function(self.x @ self.w)
        pred = np.clip(pred, 1e-15, 1 - 1e-15)  # para evitar log(0)
        grad : float  = -np.sum(self.b * np.log(pred) + ((1 - self.b) * np.log(1 - pred)))
        termL2 : float = self.L2 * (self.w.T @ self.w)
        return grad + termL2

    def gradiente_cross_entropy(self) -> np.ndarray:
        pred : float = self.sigmoid_function(self.x @ self.w)
        pred = np.clip(pred, 1e-15, 1 - 1e-15)  # para evitar log(0)
        grad : np.ndarray = -self.x.T @ (self.b - pred)
        termL2 : np.ndarray = 2 * self.L2 * self.w
        return grad + termL2

    def predict(self, input : np.ndarray) -> None:
        input_with_bias : np.ndarray = np.array(np.c_[np.ones(input.shape[0]), input], dtype=np.float64)   # agrego columna de unos para el bias.
        self.pred_probs = self.sigmoid_function(input_with_bias @ self.w)

    def evaluate(self, ground_truth : np.ndarray, threshold : float = 0.5) -> None:
        pred : np.ndarray[int] = (self.pred_probs > threshold).astype(int)
        self.tp = np.sum((pred == 1) & (ground_truth == 1))
        self.tn = np.sum((pred == 0) & (ground_truth == 0))
        self.fp = np.sum((pred == 1) & (ground_truth == 0))
        self.fn = np.sum((pred == 0) & (ground_truth == 1))

    def get_confusion_matrix(self) -> tuple[int, int, int, int]:
        return (self.tp, self.tn, self.fp, self.fn)
    
    def get_accuracy(self) -> float:
        if (self.tp + self.tn + self.fp + self.fn) == 0:
            return 0.0
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def get_false_positive_rate(self) -> float:
        if (self.fp + self.tn) == 0:
            return 0.0
        return (self.fp) / (self.fp + self.tn)
    
    def get_precision(self) -> float:
        if (self.tp + self.fp) == 0:
            return 0.0
        return (self.tp) / (self.tp + self.fp)

    def get_recall(self) -> float:  # También True Positive Rate
        if (self.tp + self.fn) == 0:
            return 0.0
        return (self.tp) / (self.tp + self.fn)
    
    def get_f_score(self) -> float:
        if ((2*self.tp) + self.fp + self.fn) == 0:
            return 0.0
        return (2*self.tp) / ((2*self.tp) + self.fp + self.fn)

    def print_weights(self, weight_names : list[str]) -> None:
        print(f'{'BIAS':14}', '(w0): ', self.w[0])
        for i in range(self.x.shape[1] - 1):
            print(f'{weight_names[i]:14} (w{i+1}): ', self.w[i+1])
    
    def print_metrics(self) -> None:
        print("ACCURACY             : ", self.get_accuracy())
        print("PRECISION            : ", self.get_precision())
        print("RECALL               : ", self.get_recall())
        print("FALSE POSITIVE RATE  : ", self.get_false_positive_rate())
        print("F-SCORE              : ", self.get_f_score())
    
    def plot_confusion_matrix(self) -> None:
        tp, tn, fp, fn = self.get_confusion_matrix()
        conf_matrix = np.array([[tp, fn],
                                [fp, tn]])
        conf_matrix_labels = ['Positive', 'Negative']
        df_cm = pd.DataFrame(conf_matrix, index=conf_matrix_labels, columns=conf_matrix_labels)
        sb.heatmap(df_cm, annot=True, fmt='d', cmap='Purples')
        plt.title('Diagnosis Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.show()
        

    def get_roc_points(self, ground_truth : np.ndarray, k_points : int = 10) -> tuple[list[float], list[float]]:
        original_tp : int = self.tp
        original_tn : int = self.tn
        original_fp : int = self.fp
        original_fn : int = self.fn
        recalls : list[float] = []
        precisions : list[float] = []
        for threshold in np.linspace(0, 1, k_points):
            self.evaluate(ground_truth, threshold=threshold)
            recalls.append(self.get_recall())
            precisions.append(self.get_precision())
        self.tp : int = original_tp
        self.tn : int = original_tn
        self.fp : int = original_fp
        self.fn : int = original_fn
        return (recalls, precisions)

    def get_pr_points(self, ground_truth : np.ndarray, k_points : int = 10) -> tuple[list[float], list[float]]:
        original_tp : int = self.tp
        original_tn : int = self.tn
        original_fp : int = self.fp
        original_fn : int = self.fn
        trues_positives_rateses : list[float] = []
        falses_positives_rateses : list[float] = []
        for threshold in np.linspace(0, 1, k_points):
            self.evaluate(ground_truth, threshold=threshold)
            trues_positives_rateses.append(self.get_recall())
            falses_positives_rateses.append(self.get_false_positive_rate())
        self.tp : int = original_tp
        self.tn : int = original_tn
        self.fp : int = original_fp
        self.fn : int = original_fn
        return (falses_positives_rateses, trues_positives_rateses)