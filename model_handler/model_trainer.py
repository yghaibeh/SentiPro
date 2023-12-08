from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import numpy as np

import torch
import logging
from typing import List, Tuple, Optional, Dict

from utils.config_utils import CustomConfig
from utils.metrics import CustomMetrics
from torch.utils.data import DataLoader

# Set up logging
logger = logging.getLogger(__name__)


class SentimentAnalysisModel:
    """
    BERT-based sentiment analysis model.

    Attributes:
        dataloader_train (torch.utils.data_handlers.DataLoader): Training data_handlers loader.
        dataloader_val (torch.utils.data_handlers.DataLoader): Validation data_handlers loader.
        dataloader_test (torch.utils.data_handlers.DataLoader): Test data_handlers loader.
        config (CustomConfig): Configuration for model training.
        scheduler (Optional[torch.optim.lr_scheduler]): Learning rate scheduler.
        optimizer (Optional[torch.optim.Optimizer]): Model optimizer.
        model (Optional[BertForSequenceClassification]): BERT model.

    Methods:
        set_model(): Initialize the BERT model.
        set_device(): Move the model to the specified device.
        set_optimizer(): Initialize the optimizer.
        set_scheduler(): Initialize the learning rate scheduler.
        save_model(): Save the model state.
        _run_epoch(): Run a single training or evaluation epoch.
        train(): Train the model for multiple epochs.
        evaluate(): Evaluate the model on the validation set.
    """

    def __init__(self,
                 dataloader_train: torch.utils.data.DataLoader,
                 dataloader_val: torch.utils.data.DataLoader,
                 dataloader_test: torch.utils.data.DataLoader,
                 config: CustomConfig) -> None:
        """
        Initialize the SentimentAnalysisModel.

        Args:
            dataloader_train (torch.utils.data_handlers.DataLoader): Training data_handlers loader.
            dataloader_val (torch.utils.data_handlers.DataLoader): Validation data_handlers loader.
            dataloader_test (torch.utils.data_handlers.DataLoader): Test data_handlers loader.
            config (CustomConfig): Configuration for model training.
        """
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.dataloader_test = dataloader_test
        self.config = config
        self.scheduler: Optional[torch.optim.lr_scheduler] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model: Optional[BertForSequenceClassification] = None

    def set_model(self, model=None) -> None:
        """
        Initialize the BERT model.
        """

        if model:
            self.model = model
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                self.config.pretrained_model,
                num_labels=self.config.num_classes,
                output_attentions=False,
                output_hidden_states=False
            )

    def set_device(self) -> None:
        """
        Move the model to the specified device.
        """
        self.model.to(self.config.device)

    def set_optimizer(self) -> None:
        """
        Initialize the optimizer.
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, eps=self.config.eps)

    def set_scheduler(self) -> None:
        """
        Initialize the learning rate scheduler.
        """
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.dataloader_train) * self.config.epochs
        )

    def save_model(self, epoch: int) -> None:
        """
        Save the model state.

        Args:
            epoch (int): Current epoch number.
        """
        model_path = f'_BERT_epoch_{epoch}.model'
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved at: {model_path}")

    def _run_epoch(
            self,
            data_loader: torch.utils.data.DataLoader,
            optimizer: Optional[torch.optim.Optimizer],
            criterion: torch.nn.Module,
            device: torch.device,
            desc: str
    ) -> Tuple[float, List[np.ndarray], np.ndarray]:
        """
        Run a single training or evaluation epoch.

        Args:
            data_loader (torch.utils.data_handlers.DataLoader): Data loader for the epoch.
            optimizer (Optional[torch.optim.Optimizer]): Model optimizer (None for evaluation).
            criterion (torch.nn.Module): Loss criterion.
            device (torch.device): Device for training/evaluation.
            desc (str): Description for tqdm.

        Returns:
            Tuple[float, List[np.ndarray], np.ndarray]: Average loss, predictions, true labels.
        """
        self.model.train() if optimizer else self.model.eval()
        total_loss = 0.0
        preds, true_labels = [], []

        with torch.set_grad_enabled(optimizer is not None):
            for i, batch in enumerate(tqdm(data_loader, desc=desc, unit="batch", leave=False, disable=not optimizer)):
                batch = tuple(b.to(device) for b in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

                outputs = self.model(**inputs)
                loss = criterion(outputs.logits, inputs['labels'])
                total_loss += loss.item()

                if optimizer:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    self.scheduler.step()
                    optimizer.zero_grad()

                preds.append(outputs.logits.detach().cpu().numpy())
                true_labels.append(inputs['labels'].detach().cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        preds = np.concatenate(preds, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)

        return avg_loss, preds, true_labels

    def train(self) -> None:
        """
        Train the model for multiple epochs.
        """
        for epoch in tqdm(range(1, self.config.epochs + 1), desc='Epochs'):
            train_loss, _, _ = self._run_epoch(
                self.dataloader_train,
                self.optimizer,
                torch.nn.CrossEntropyLoss(),
                self.config.device,
                f'Training Epoch {epoch}'
            )
            val_loss, predictions, true_vals = self.evaluate()
            val_f1 = CustomMetrics.compute_f1_score(predictions, true_vals)

            logger.info(f'\nEpoch {epoch}')
            logger.info(f'Training loss: {train_loss}')
            logger.info(f'Validation loss: {val_loss}')
            logger.info(f'F1 Score (Weighted): {val_f1}')

            self.save_model(epoch)

    def evaluate(self) -> Tuple[float, List[np.ndarray], np.ndarray]:
        """
        Evaluate the model on the validation set and print the classification report.

        Returns:
            Tuple[float, List[np.ndarray], np.ndarray]: Average loss, predictions, true labels.
        """
        val_loss, predictions, true_vals = self._run_epoch(
            self.dataloader_val,
            None,
            torch.nn.CrossEntropyLoss(),
            self.config.device,
            "Evaluating"
        )

        CustomMetrics.print_classification_report(predictions, true_vals)

        return val_loss, predictions, true_vals

    def test(self) -> Tuple[float, List[np.ndarray], np.ndarray]:
        """
        Test the model on the test set and print the classification report.

        Returns:
            Tuple[float, List[np.ndarray], np.ndarray]: Average loss, predictions, true labels.
        """
        test_loss, predictions, true_vals = self._run_epoch(
            self.dataloader_test,
            None,
            torch.nn.CrossEntropyLoss(),
            self.config.device,
            "Testing"
        )

        self.print_classification_report(predictions, true_vals)

        return test_loss, predictions, true_vals

    def load_and_evaluate_on_saved_model(self, model_path: str, label_dict: Dict[int, str]) -> Tuple[
        float, List[np.ndarray], np.ndarray]:
        """
        Load a saved model and evaluate it on the validation and test sets.

        Args:
            model_path (str): Path to the saved model.
            label_dict (Dict[int, str]): Mapping of class indices to labels.

        Returns:
            Tuple[float, List[np.ndarray], np.ndarray]: Test loss, predictions, true labels.
        """
        # Load the model
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        logger.info(f"Model loaded from: {model_path}")

        # Evaluate on the validation set
        val_loss, val_predictions, val_true_labels = self.evaluate()
        val_f1 = CustomMetrics.compute_f1_score(val_predictions, val_true_labels)

        logger.info(f'Validation loss: {val_loss}')
        logger.info(f'F1 Score (Weighted): {val_f1}')

        # Test the model on the test set
        test_loss, test_predictions, test_true_labels = self._run_epoch(
            self.dataloader_test,
            None,
            torch.nn.CrossEntropyLoss(),
            self.config.device,
            "Testing"
        )
        logger.info('--------------')

        logger.info(f'Test loss: {test_loss}')

        # Additional logs for test results
        test_f1 = CustomMetrics.compute_f1_score(test_predictions, test_true_labels)
        overall_accuracy, class_accuracies = CustomMetrics.compute_accuracy(test_predictions, test_true_labels,
                                                                            label_dict)

        logger.info(f'Test F1 Score (Weighted): {test_f1}')
        logger.info(f'Test Accuracy: {overall_accuracy:.4f}')
        for class_label, class_acc in class_accuracies.items():
            print(f'Class: {class_label} - Accuracy: {class_acc:.4f}')

        return test_loss, test_predictions, test_true_labels
