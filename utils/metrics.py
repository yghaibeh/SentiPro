from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import numpy as np

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class CustomMetrics:
    """
    Class for custom evaluation metrics.

    Methods:
        f1_score_func(): Compute the weighted F1 score.
        accuracy_per_class(): Print accuracy per class.
    """

    @staticmethod
    def compute_f1_score(preds: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the weighted F1 score.

        Args:
            preds (np.ndarray): Predicted labels.
            labels (np.ndarray): True labels.

        Returns:
            float: Weighted F1 score.
        """
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    @staticmethod
    def compute_accuracy(preds: np.ndarray, labels: np.ndarray, label_dict: Dict[int, str]) -> Tuple[
        float, Dict[str, float]]:
        """
        Calculate accuracy per class.

        Args:
            preds (np.ndarray): Predicted labels.
            labels (np.ndarray): True labels.
            label_dict (Dict[int, str]): Mapping of class indices to labels.

        Returns:
            Tuple[float, Dict[str, float]]: Overall accuracy and class-wise accuracy.
        """

        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        correct_predictions = np.sum(preds_flat == labels_flat)
        total_predictions = len(labels_flat)

        overall_accuracy = correct_predictions / total_predictions
        class_accuracies = {}

        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            class_accuracy = np.sum(y_preds == label) / len(y_true)
            class_accuracies[label_dict[label]] = class_accuracy

        return overall_accuracy, class_accuracies

    @staticmethod
    def print_classification_report(predictions: np.ndarray, true_vals: np.ndarray) -> None:
        """
        Print classification report using sklearn.metrics.classification_report.

        Args:
            predictions (np.ndarray): Model predictions.
            true_vals (np.ndarray): True labels.
        """
        preds_flat = np.argmax(predictions, axis=1).flatten()
        report = classification_report(preds_flat, true_vals)
        print(report)

