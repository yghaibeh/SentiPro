from transformers import BertForSequenceClassification, BertTokenizer
import torch
import logging
from typing import List, Tuple
import numpy as np

from utils.config_utils import CustomConfig

# Set up logging
logger = logging.getLogger(__name__)


class SentimentAnalysisInference:
    def __init__(self, model_path: str, config: CustomConfig):
        """
        Initialize the SentimentAnalysisInference class.

        Args:
            model_path (str): Path to the saved pre-trained model.
            config (CustomConfig): Configuration for model inference.
        """
        self.model_path: str = model_path
        self.config: CustomConfig = config
        self.model: BertForSequenceClassification = self._load_model()

    def _load_model(self) -> BertForSequenceClassification:
        """
        Load the pre-trained model.

        Returns:
            BertForSequenceClassification: Loaded pre-trained model.
        """

        try:
            model = BertForSequenceClassification.from_pretrained(
                self.config.pretrained_model,
                num_labels=self.config.num_classes,
                output_attentions=False,
                output_hidden_states=False
            )
            m = torch.load(self.model_path, map_location=torch.device('cpu'))
            model.load_state_dict(m)
            model.eval()
            return model
        except Exception as e:
            logger.error('Error loading the model: {}'.format(str(e)))

    def infer(self, text_data: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        Perform inference on new data_handlers.

        Args:
            text_data (List[str]): List of text data_handlers to perform inference on.

        Returns:
            Tuple[np.ndarray, List[int]]: Predicted probabilities and corresponding labels.
        """
        tokenizer = BertTokenizer.from_pretrained(self.config.pretrained_model, do_lower_case=True)
        encoded_data = tokenizer.batch_encode_plus(
            text_data,
            add_special_tokens=self.config.add_special_tokens,
            return_attention_mask=self.config.return_attention_mask,
            pad_to_max_length=self.config.pad_to_max_length,
            max_length=self.config.max_length,
            return_tensors=self.config.return_tensors
        )

        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']

        with torch.no_grad():
            inputs = {'input_ids': input_ids, 'attention_mask': attention_masks}
            outputs = self.model(**inputs)

        logits = outputs.logits.numpy()
        probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1).numpy()
        predicted_labels = np.argmax(probabilities, axis=1).tolist()

        return probabilities, predicted_labels

    def infer_with_percentages(self, text: str) -> dict:
        """
        Perform sentiment analysis on the given text and return percentages for each sentiment class.

        Args:
            text (str): Text to perform sentiment analysis on.

        Returns:
            dict: A dictionary containing sentiment analysis results.
                  - 'probabilities': A list of integers representing the percentages for each sentiment class.
                  - 'predicted_label': The predicted sentiment label ('Negative', 'Neutral', or 'Positive').
        """
        # Perform inference and obtain probabilities and predicted labels
        probabilities, predicted_label = self.infer([text])

        # Extract probabilities for the predicted label and convert them to percentages
        probabilities = probabilities[0]
        percentages = [round(prob * 100, 1) for prob in probabilities]

        # Extract the predicted label and map it to a sentiment category
        predicted_label = predicted_label[0]
        labels_map = {
            0: 'Negative',
            1: 'Neutral',
            2: 'Positive'
        }
        predicted_sentiment = labels_map[predicted_label]

        result_dict = {'probabilities': percentages, 'predicted_label': predicted_sentiment}
        return result_dict

