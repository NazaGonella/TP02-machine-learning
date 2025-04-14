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
        plt.title('War Class Confusion Matrix')
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

class LinearDiscriminantAnalysis:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x: np.ndarray = np.array(x, dtype=np.float64)
        self.y: np.ndarray = np.array(y, dtype=np.int32)
        self.classes: np.ndarray = np.unique(self.y)
        self.means: dict[int, np.ndarray] = {}
        self.priors: dict[int, float] = {}
        self.covariance: np.ndarray = np.array([])
        self.inv_covariance: np.ndarray = np.array([])
        self.fitted: bool = False
        self.pred_labels: np.ndarray = np.array([])

    def fit(self) -> None:
        n_samples, n_features = self.x.shape
        self.covariance = np.zeros((n_features, n_features))

        for c in self.classes:
            x_c = self.x[self.y == c]
            self.means[c] = np.mean(x_c, axis=0)
            self.priors[c] = x_c.shape[0] / n_samples
            self.covariance += np.cov(x_c, rowvar=False, bias=True) * x_c.shape[0]

        self.covariance /= n_samples
        self.inv_covariance = np.linalg.inv(self.covariance)
        self.fitted = True

    def _discriminant_function(self, x: np.ndarray, c: int) -> float:
        mu = self.means[c]
        prior = self.priors[c]
        return float(x @ self.inv_covariance @ mu.T - 0.5 * mu.T @ self.inv_covariance @ mu + np.log(prior))

    def predict(self, input: np.ndarray) -> None:
        input = np.array(input, dtype=np.float64)
        pred = []
        for x_i in input:
            scores = {c: self._discriminant_function(x_i, c) for c in self.classes}
            pred.append(max(scores, key=scores.get))
        self.pred_labels = np.array(pred, dtype=np.int32)

    def evaluate(self, ground_truth: np.ndarray) -> float:
        if self.pred_labels.size == 0:
            raise RuntimeError("Debe ejecutar predict() antes de evaluar.")
        return np.mean(self.pred_labels == ground_truth)

    def get_confusion_matrix(self, ground_truth: np.ndarray) -> np.ndarray:
        if self.pred_labels.size == 0:
            raise RuntimeError("Debe ejecutar predict() antes de generar la matriz de confusión.")
        
        label_to_index = {label: i for i, label in enumerate(self.classes)}
        n_classes = len(self.classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        for true, pred in zip(ground_truth, self.pred_labels):
            i = label_to_index[int(true)]
            j = label_to_index[int(pred)]
            matrix[i][j] += 1

        return matrix

    def get_precision(self, conf_matrix: np.ndarray) -> np.ndarray:
        precision_scores = []
        for i in range(conf_matrix.shape[0]):
            TP = conf_matrix[i, i]
            FP = np.sum(conf_matrix[:, i]) - TP
            precision_scores.append(TP / (TP + FP) if TP + FP > 0 else 0.0)
        return np.array(precision_scores)

    def get_recall(self, conf_matrix: np.ndarray) -> np.ndarray:
        recall_scores = []
        for i in range(conf_matrix.shape[0]):
            TP = conf_matrix[i, i]
            FN = np.sum(conf_matrix[i, :]) - TP
            recall_scores.append(TP / (TP + FN) if TP + FN > 0 else 0.0)
        return np.array(recall_scores)
    
    def get_f_score(self, conf_matrix: np.ndarray) -> np.ndarray:
        precision = self.get_precision(conf_matrix)
        recall = self.get_recall(conf_matrix)
        return 2 * (precision * recall) / (precision + recall)
    
    def print_metrics(self, ground_truth: np.ndarray) -> None:
        if self.pred_labels.size == 0:
            raise RuntimeError("Debe ejecutar predict() antes de evaluar.")
        
        # Get the confusion matrix
        conf_matrix = self.get_confusion_matrix(ground_truth)

        # Calculate precision, recall, and F-score
        precision = self.get_precision(conf_matrix)
        recall = self.get_recall(conf_matrix)
        f_score = self.get_f_score(conf_matrix)
        
        # Calculate accuracy
        accuracy = self.evaluate(ground_truth)

        # Print the metrics
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nAccuracy: {:.4f}".format(accuracy))
        
        for i, class_label in enumerate(self.classes):
            print(f"\nClass {class_label}:")
            print(f"  Precision: {precision[i]:.4f}")
            print(f"  Recall: {recall[i]:.4f}")
            print(f"  F-Score: {f_score[i]:.4f}")
        
        # You can add other metrics here, like AUC-ROC, AUC-PR, etc., if needed

# Testing
if __name__ == "__main__":
    # Imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sb
    import os
    import preprocessing as prepro
    import data_handler
    from IPython.display import display

    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    war_class_dev : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema2/data/raw/WAR_class_dev.csv')
    war_class_test : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema2/data/raw/WAR_class_test.csv')

    war_class_dev_processed_and_standardized : pd.DataFrame = prepro.process_and_stardardize(
        war_class_dev, 
        filename='war_class_dev', 
        save_path=f'{project_root}/TP02/problema2/data/processed/'
    )

    train : pd.DataFrame
    validation : pd.DataFrame
    train, validation = data_handler.get_train_and_validation_sets(war_class_dev_processed_and_standardized, train_fraction=0.8, seed=42)

    lda = LinearDiscriminantAnalysis(train.drop(columns=['war_class']).to_numpy(), train['war_class'].to_numpy())
    lda.fit()
    lda.predict(validation.drop(columns=['war_class']).to_numpy())
    accuracy = lda.evaluate(validation["war_class"].to_numpy())
    conf_matrix = lda.get_confusion_matrix(validation["war_class"].to_numpy())
    # Print the accuracy
    # print(f"Accuracy: {accuracy}")
    # print("Precision: ", lda.get_precision(conf_matrix))
    # print("Recall: ", lda.get_recall(conf_matrix))
    # print("F-Score: ", lda.get_f_score(conf_matrix))
    lda.print_metrics(validation["war_class"].to_numpy())

    # Plotting the confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sb.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=lda.classes, yticklabels=lda.classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()