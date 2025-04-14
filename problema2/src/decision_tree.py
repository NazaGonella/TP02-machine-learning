from collections import Counter
import numpy as np


def entropy(y):
    y = y.astype(int)
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if self.n_feats is None else min(self.n_feats, X.shape[1])
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        if (
            depth >= self.max_depth
            or len(unique_labels) == 1
            or num_samples < self.min_samples_split
        ):
            return Node(value=self._majority_class(y))

        feature_indices = np.random.choice(num_features, self.n_feats, replace=False)

        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)

        left_idx, right_idx = self._partition(X[:, best_feature], best_threshold)

        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _find_best_split(self, X, y, feature_indices):
        best_gain = -np.inf
        split_feature = None
        split_threshold = None

        for idx in feature_indices:
            values = np.unique(X[:, idx])
            for val in values:
                gain = self._calculate_information_gain(y, X[:, idx], val)

                if gain > best_gain:
                    best_gain = gain
                    split_feature = idx
                    split_threshold = val

        return split_feature, split_threshold

    def _calculate_information_gain(self, y, feature_column, threshold):
        parent_entropy = entropy(y)

        left_idx, right_idx = self._partition(feature_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        n_total = len(y)
        n_left = len(left_idx)
        n_right = len(right_idx)

        entropy_left = entropy(y[left_idx])
        entropy_right = entropy(y[right_idx])

        weighted_entropy = (n_left / n_total) * entropy_left + (n_right / n_total) * entropy_right

        return parent_entropy - weighted_entropy

    def _partition(self, feature_column, threshold):
        left_idx = np.where(feature_column <= threshold)[0]
        right_idx = np.where(feature_column > threshold)[0]
        return left_idx, right_idx

    def _predict_sample(self, x, node):
        while not node.is_leaf_node():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _majority_class(self, y):
        label_counts = Counter(y)
        return label_counts.most_common(1)[0][0]
