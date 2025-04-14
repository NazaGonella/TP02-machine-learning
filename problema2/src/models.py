import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from decision_tree import DecisionTree

class LogisticRegression:
    def __init__(self, x: np.ndarray, b: np.ndarray, L2: float = 0, initial_weight_value: float = 1):
        self.x = np.array(np.c_[np.ones(x.shape[0]), x], dtype=np.float64)
        self.L2 = L2
        self.classes = np.unique(b)
        self.num_classes = len(self.classes)
        self.b = b
        self.models = {
            cls: {
                'w': np.full(shape=self.x.shape[1], fill_value=initial_weight_value, dtype=np.float64),
                'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
                'pred_probs': np.array([])
            } for cls in self.classes
        }

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, pred, target, w):
        pred = np.clip(pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(target * np.log(pred) + (1 - target) * np.log(1 - pred))
        return loss + self.L2 * (w.T @ w)

    def gradiente_cross_entropy(self, x, pred, target, w):
        grad = -x.T @ (target - pred)
        return grad + 2 * self.L2 * w

    def fit_gradient_descent(self, step_size: float, tolerance: float = -1, max_number_of_steps: int = -1):
        for cls in self.classes:
            y_binary = (self.b == cls).astype(np.float64)
            w = self.models[cls]['w']
            attempts = 0
            while True:
                pred = self.sigmoid_function(self.x @ w)
                # print(self.binary_cross_entropy(pred=pred, target=y_binary, w=w))
                grad = self.gradiente_cross_entropy(self.x, pred, y_binary, w)
                if (np.linalg.norm(grad) <= tolerance and tolerance != -1) or \
                   (attempts >= max_number_of_steps and max_number_of_steps != -1):
                    break
                w -= step_size * grad
                attempts += 1
            self.models[cls]['w'] = w

    def predict(self, input: np.ndarray):
        input_with_bias = np.array(np.c_[np.ones(input.shape[0]), input], dtype=np.float64)
        probs = {}
        for cls in self.classes:
            w = self.models[cls]['w']
            probs[cls] = self.sigmoid_function(input_with_bias @ w)
            self.models[cls]['pred_probs'] = probs[cls]
        stacked_probs = np.stack([probs[cls] for cls in self.classes], axis=1)
        return self.classes[np.argmax(stacked_probs, axis=1)]

    def evaluate(self, ground_truth: np.ndarray, input: np.ndarray, threshold: float = 0.5):
        pred = self.predict(input)  # input debe ser el conjunto a evaluar
        for cls in self.classes:
            gt_binary = (ground_truth == cls).astype(int)
            pred_binary = (pred == cls).astype(int)
            self.models[cls]['tp'] = np.sum((pred_binary == 1) & (gt_binary == 1))
            self.models[cls]['tn'] = np.sum((pred_binary == 0) & (gt_binary == 0))
            self.models[cls]['fp'] = np.sum((pred_binary == 1) & (gt_binary == 0))
            self.models[cls]['fn'] = np.sum((pred_binary == 0) & (gt_binary == 1))

    def get_accuracy(self) -> float:
        pred = self.predict(self.x[:, 1:])
        return np.mean(pred == self.b)

    def get_confusion_matrix(self) -> np.ndarray:
        pred = self.predict(self.x[:, 1:])
        conf_matrix = pd.crosstab(self.b, pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
        return conf_matrix

    def print_metrics(self):
        print("Total Accuracy :", self.get_accuracy())
        for cls in self.classes:
            tp = self.models[cls]['tp']
            fp = self.models[cls]['fp']
            fn = self.models[cls]['fn']
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"\nClass {cls}:")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall   : {rec:.4f}")
            print(f"  F1 Score : {f1:.4f}")

    def plot_confusion_matrix(self):
        conf_matrix = self.get_confusion_matrix()
        sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples')
        plt.title('Multiclass Confusion Matrix')
        plt.show()
    
    

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
        self.scores: np.ndarray = np.array([])
        # self.metrics_per_class : dict[int, tuple[int, int, int, int]] = {c : (0,0,0,0) for c in self.classes}
        self.metrics: dict[int, list[int, int, int, int]] = {c : [0,0,0,0] for c in self.classes}
        self.metrics_threshold: dict[int, list[int, int, int, int]] = {c : [0,0,0,0] for c in self.classes}


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
        all_scores = []

        for x_i in input:
            scores = {c: self._discriminant_function(x_i, c) for c in self.classes}
            all_scores.append(scores)
            pred.append(max(scores, key=scores.get))

        self.pred_labels = np.array(pred, dtype=np.int32)
        self.scores = np.array(all_scores)  # array de diccionarios de scores

    def evaluate(self, ground_truth: np.ndarray) -> float:
        if self.pred_labels.size == 0:
            raise RuntimeError("Debe ejecutar predict() antes de evaluar.")
        
        for c in self.classes:
            y_true = (ground_truth == c).astype(int)
            y_pred = (self.pred_labels == c).astype(int)

            TP = np.sum((y_pred == 1) & (y_true == 1))
            TN = np.sum((y_pred == 0) & (y_true == 0))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            FN = np.sum((y_pred == 0) & (y_true == 1))
            self.metrics[c] = [TP, TN, FP, FN]
        return np.mean(self.pred_labels == ground_truth)

    def evaluate_threshold(self, ground_truth: np.ndarray, threshold: float):
        if self.scores.size == 0:
            raise RuntimeError("Debe ejecutar predict() antes de evaluar con umbral.")

        for c in self.classes:
            y_true = (ground_truth == c).astype(int)
            # y_scores = np.array([score[c] for score in self.scores])
            y_scores = np.array([score[c] for score in self.scores])
            y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)

            y_pred = (y_scores >= threshold).astype(int)

            TP = np.sum((y_pred == 1) & (y_true == 1))
            TN = np.sum((y_pred == 0) & (y_true == 0))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            FN = np.sum((y_pred == 0) & (y_true == 1))
            self.metrics_threshold[c] = [TP, TN, FP, FN]

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
    
    def get_accuracy(self, class_label: int, from_threshold : bool = True) -> float:
        TP = self.metrics_threshold[class_label][0] if from_threshold else self.metrics[class_label][0]
        TN = self.metrics_threshold[class_label][1] if from_threshold else self.metrics[class_label][1]
        FP = self.metrics_threshold[class_label][2] if from_threshold else self.metrics[class_label][2]
        FN = self.metrics_threshold[class_label][3] if from_threshold else self.metrics[class_label][3]
        return (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    def get_precision(self, class_label: int, from_threshold : bool = True) -> float:
        TP = self.metrics_threshold[class_label][0] if from_threshold else self.metrics[class_label][0]
        FP = self.metrics_threshold[class_label][2] if from_threshold else self.metrics[class_label][2]
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    def get_recall(self, class_label: int, from_threshold : bool = True) -> float:
        TP = self.metrics_threshold[class_label][0] if from_threshold else self.metrics[class_label][0]
        FN = self.metrics_threshold[class_label][3] if from_threshold else self.metrics[class_label][3]
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def get_false_positive_rate(self, class_label: int, from_threshold : bool = True) -> float:
        TP = self.metrics_threshold[class_label][0] if from_threshold else self.metrics[class_label][0]
        TN = self.metrics_threshold[class_label][1] if from_threshold else self.metrics[class_label][1]
        FP = self.metrics_threshold[class_label][2] if from_threshold else self.metrics[class_label][2]
        FN = self.metrics_threshold[class_label][3] if from_threshold else self.metrics[class_label][3]
        return FP / (FP + TN) if (FP + TN) > 0 else 0.0
    
    def get_f_score(self, class_label : int, from_threshold : bool = True) -> np.ndarray:
        precision = self.get_precision(class_label=class_label, from_threshold=from_threshold)
        recall = self.get_recall(class_label=class_label, from_threshold=from_threshold)
        return 2 * (precision * recall) / (precision + recall)
    
    def get_roc_points(self, ground_truth : np.ndarray, class_label : int, k_points : int = 10) -> tuple[list[float], list[float]]:
        original_tp : int = self.metrics_threshold[class_label][0]
        original_tn : int = self.metrics_threshold[class_label][1] 
        original_fp : int = self.metrics_threshold[class_label][2] 
        original_fn : int = self.metrics_threshold[class_label][3] 
        recalls : list[float] = []
        precisions : list[float] = []
        for threshold in np.linspace(0, 1, k_points):
            self.evaluate_threshold(ground_truth, threshold=threshold)
            recalls.append(self.get_recall(class_label=class_label, from_threshold=True))
            precisions.append(self.get_precision(class_label=class_label, from_threshold=True))
        self.metrics_threshold[class_label][0] = original_tp
        self.metrics_threshold[class_label][1] = original_tn
        self.metrics_threshold[class_label][2] = original_fp
        self.metrics_threshold[class_label][3] = original_fn
        return (recalls, precisions)
    
    def get_pr_points(self, ground_truth : np.ndarray, class_label : int, k_points : int = 10) -> tuple[list[float], list[float]]:
        original_tp : int = self.metrics_threshold[class_label][0]
        original_tn : int = self.metrics_threshold[class_label][1] 
        original_fp : int = self.metrics_threshold[class_label][2] 
        original_fn : int = self.metrics_threshold[class_label][3] 
        trues_positives_rateses : list[float] = []
        falses_positives_rateses : list[float] = []
        for threshold in np.linspace(0, 1, k_points):
            self.evaluate_threshold(ground_truth, threshold=threshold)
            trues_positives_rateses.append(self.get_recall(class_label=class_label, from_threshold=True))
            falses_positives_rateses.append(self.get_false_positive_rate(class_label=class_label, from_threshold=True))
        self.metrics_threshold[class_label][0] = original_tp
        self.metrics_threshold[class_label][1] = original_tn
        self.metrics_threshold[class_label][2] = original_fp
        self.metrics_threshold[class_label][3] = original_fn
        return (falses_positives_rateses, trues_positives_rateses)

    def print_metrics(self, ground_truth: np.ndarray) -> None:
        if self.pred_labels.size == 0:
            raise RuntimeError("Debe ejecutar predict() antes de evaluar.")
        for label in self.classes:
            print(f"\nClass {label}:")
            print(f"Precision: {self.get_precision(class_label=label, from_threshold=False):.4f}")
            print(f"Recall: {self.get_recall(class_label=label, from_threshold=False):.4f}")
            print(f"F-Score: {self.get_f_score(class_label=label, from_threshold=False):.4f}")
            recalls, precisions = self.get_roc_points(validation["war_class"].to_numpy(), label, 20)
            sorted_x : np.ndarray[float] = np.sort(recalls)
            sorted_y : np.ndarray[float] = np.sort(precisions)
            falses_positives_rateses, trues_positives_rateses = self.get_pr_points(validation["war_class"].to_numpy(), label, 20)
            print(f"AUC-ROC: {np.trapz(y=recalls, x=precisions):.4f}")
            sorted_x = np.sort(falses_positives_rateses)
            sorted_y = np.sort(trues_positives_rateses)
            print(f"AUC-PR: {np.trapz(y=sorted_y, x=sorted_x):.4f}")

    def plot_confusion_matrix(self, conf_matrix) -> None:
        # Plotting the confusion matrix using seaborn heatmap
        plt.figure(figsize=(8, 6))
        sb.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=self.classes, yticklabels=self.classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def plot_roc_curve(self) -> None:
        for c in self.classes:
            recalls, precisions = self.get_roc_points(validation["war_class"].to_numpy(), c, 20)
            plt.plot(recalls, precisions, label=f'Class {c}')
        plt.title("ROC")
        plt.grid(visible=True, alpha=0.5)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()

    def plot_pr_curve(self) -> None:
        for c in self.classes:
            falses_positives_rateses, trues_positives_rateses = self.get_pr_points(validation["war_class"].to_numpy(), c, 20)
            plt.plot(falses_positives_rateses, trues_positives_rateses, label=f'Class {c}')
        plt.title("Precision-Recall (PR)")
        plt.grid(visible=True, alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
        self.fitted = False
        self.pred_labels = np.array([])
        self.metrics = {}
        self.classes = None

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def most_common_label(self, y):
        label_counts = {}
        for label in y:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        most_common = max(label_counts, key=label_counts.get)
        return most_common

    def fit(self, X, y):
        self.trees = []
        self.classes = np.unique(y)
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            X_samp, y_samp = self.bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("El modelo no ha sido ajustado. Ejecute 'fit' primero.")
        
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        self.pred_labels = np.array([self.most_common_label(tree_pred) for tree_pred in tree_preds])
        return self.pred_labels

    def evaluate(self, ground_truth: np.ndarray) -> float:
        if self.pred_labels.size == 0:
            raise RuntimeError("Debe ejecutar 'predict()' antes de evaluar.")
        accuracy = np.mean(self.pred_labels == ground_truth)
        return accuracy

    def get_confusion_matrix(self, ground_truth: np.ndarray) -> np.ndarray:
        if self.pred_labels.size == 0:
            raise RuntimeError("Debe ejecutar 'predict()' antes de generar la matriz de confusión.")
        
        classes = np.unique(ground_truth)
        label_to_index = {label: i for i, label in enumerate(classes)}
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        for true, pred in zip(ground_truth, self.pred_labels):
            i = label_to_index[int(true)]
            j = label_to_index[int(pred)]
            matrix[i][j] += 1

        return matrix

    def get_accuracy(self, class_label: int) -> float:
        TP = self.metrics[class_label][0]
        TN = self.metrics[class_label][1]
        FP = self.metrics[class_label][2]
        FN = self.metrics[class_label][3]
        return (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    def get_precision(self, class_label: int) -> float:
        TP = self.metrics[class_label][0]
        FP = self.metrics[class_label][2]
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    def get_recall(self, class_label: int) -> float:
        TP = self.metrics[class_label][0]
        FN = self.metrics[class_label][3]
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def get_f_score(self, class_label: int) -> float:
        precision = self.get_precision(class_label)
        recall = self.get_recall(class_label)
        return 2 * (precision * recall) / (precision + recall)

    def print_metrics(self, ground_truth: np.ndarray) -> None:
        accuracy = self.evaluate(ground_truth)
        print(f"Accuracy: {accuracy:.4f}")

        # Initialize the metrics for each class
        for class_label in np.unique(ground_truth):
            TP = TN = FP = FN = 0
            for true, pred in zip(ground_truth, self.pred_labels):
                if true == class_label and pred == class_label:
                    TP += 1
                elif true != class_label and pred != class_label:
                    TN += 1
                elif true != class_label and pred == class_label:
                    FP += 1
                elif true == class_label and pred != class_label:
                    FN += 1
            
            self.metrics[class_label] = [TP, TN, FP, FN]
            print(f"\nClass {class_label}:")
            print(f"Precision: {self.get_precision(class_label):.4f}")
            print(f"Recall: {self.get_recall(class_label):.4f}")
            print(f"F-Score: {self.get_f_score(class_label):.4f}")

    def plot_confusion_matrix(self, conf_matrix) -> None:
        # Plotting the confusion matrix using seaborn heatmap
        plt.figure(figsize=(8, 6))
        sb.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=self.classes, yticklabels=self.classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()



        

# Testing
# if __name__ == "__main__":
#     # Imports
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sb
#     import os
#     import preprocessing as prepro
#     import data_handler
#     from IPython.display import display

#     project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

#     war_class_dev : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema2/data/raw/WAR_class_dev.csv')
#     war_class_test : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema2/data/raw/WAR_class_test.csv')

#     war_class_dev_processed_and_standardized : pd.DataFrame = prepro.process_and_stardardize(
#         war_class_dev, 
#         filename='war_class_dev', 
#         save_path=f'{project_root}/TP02/problema2/data/processed/'
#     )

#     train : pd.DataFrame
#     validation : pd.DataFrame
#     train, validation = data_handler.get_train_and_validation_sets(war_class_dev_processed_and_standardized, train_fraction=0.8, seed=42)

#     lda = LinearDiscriminantAnalysis(train.drop(columns=['war_class']).to_numpy(), train['war_class'].to_numpy())
#     lda.fit()
#     lda.predict(validation.drop(columns=['war_class']).to_numpy())
#     total_accuracy : float = lda.evaluate(validation['war_class'].to_numpy())
#     lda.evaluate_threshold(validation['war_class'].to_numpy(), threshold=0.5)
#     print("Total Accuracy: ", total_accuracy)
#     lda.print_metrics(validation["war_class"].to_numpy())
#     lda.plot_confusion_matrix(lda.get_confusion_matrix(validation["war_class"].to_numpy()))
#     lda.plot_roc_curve()
#     lda.plot_pr_curve()

# if __name__ == "__main__":
#     # Imports
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sb
#     import os
#     import preprocessing as prepro
#     import data_handler
#     from IPython.display import display

#     project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

#     war_class_dev : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema2/data/raw/WAR_class_dev.csv')
#     war_class_test : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema2/data/raw/WAR_class_test.csv')

#     war_class_dev_processed_and_standardized : pd.DataFrame = prepro.process_and_stardardize(
#         war_class_dev, 
#         filename='war_class_dev', 
#         save_path=f'{project_root}/TP02/problema2/data/processed/'
#     )

#     train : pd.DataFrame
#     validation : pd.DataFrame
#     train, validation = data_handler.get_train_and_validation_sets(war_class_dev_processed_and_standardized, train_fraction=0.8, seed=42)

#     log_reg : LogisticRegression = LogisticRegression(train.drop(columns=['war_class']).to_numpy(), train['war_class'].to_numpy(), L2=0)
#     log_reg.fit_gradient_descent(step_size=0.001, tolerance=0.001, max_number_of_steps=10000)
#     total_accuracy : float = log_reg.get_accuracy()
#     # log_reg.evaluate(validation['war_class'].to_numpy())
#     log_reg.evaluate(ground_truth=validation['war_class'].to_numpy(), input=validation.drop(columns=['war_class']).to_numpy())
#     log_reg.print_metrics()
#     log_reg.plot_confusion_matrix()

# if __name__ == "__main__":
#     # Imports
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sb
#     import os
#     import preprocessing as prepro
#     import data_handler
#     from IPython.display import display

#     project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

#     war_class_dev : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema2/data/raw/WAR_class_dev.csv')
#     war_class_test : pd.DataFrame = pd.read_csv(f'{project_root}/TP02/problema2/data/raw/WAR_class_test.csv')

#     war_class_dev_processed_and_standardized : pd.DataFrame = prepro.process_and_stardardize(
#         war_class_dev, 
#         filename='war_class_dev', 
#         save_path=f'{project_root}/TP02/problema2/data/processed/'
#     )

#     train : pd.DataFrame
#     validation : pd.DataFrame
#     train, validation = data_handler.get_train_and_validation_sets(war_class_dev_processed_and_standardized, train_fraction=0.8, seed=42)

#     # Train RandomForest model
#     rf = RandomForest(n_trees=10, min_samples_split=2, max_depth=10)
#     rf.fit(train.drop(columns=['war_class']).to_numpy(), train['war_class'].to_numpy())
#     print("LISTO")

#     # Predict and evaluate the model
#     rf.predict(validation.drop(columns=['war_class']).to_numpy())
#     total_accuracy : float = rf.evaluate(validation['war_class'].to_numpy())
#     print("Total Accuracy: ", total_accuracy)
#     rf.print_metrics(validation["war_class"].to_numpy())

#     # # Plot confusion matrix
#     rf.plot_confusion_matrix(rf.get_confusion_matrix(validation["war_class"].to_numpy()))
