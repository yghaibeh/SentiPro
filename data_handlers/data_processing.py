import logging
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import string
from sklearn.utils import resample
from torch.utils.data import TensorDataset
from typing import Union, List
from utils.config_utils import CustomConfig

# Set up logging
logger = logging.getLogger(__name__)


class CustomTextCleaner:
    """Clears text except for digits, parentheses, dots, commas, and word characters."""

    def __init__(self, clean_pattern: str = r"[^A-ZĞÜŞİÖÇIa-zğüı'şöç0-9.\"',()]") -> None:
        """
        Initialize the CustomTextCleaner with a clean pattern.

        Args:
            clean_pattern (str): Regular expression pattern for cleaning text.
        """
        self.clean_pattern = clean_pattern

    def __call__(self, text: Union[str, List[str]]) -> List[List[str]]:
        """
        Clean the input text.

        Args:
            text (Union[str, List[str]]): Input text or list of texts.

        Returns:
            List[List[str]]: Cleaned text.
        """
        elements = None

        if isinstance(text, str):
            elements = [[text]]

        if isinstance(text, list):
            elements = text

        cleaned_text = [[re.sub(self.clean_pattern, " ", item) for item in items] for items in elements]

        return cleaned_text


class TextProcessing:
    """Class for text processing functions"""

    @staticmethod
    def emoji_remover(data: str) -> str:
        """
        Remove emojis from the input text.

        Args:
            data (str): Input text.

        Returns:
            str: Text with emojis removed.
        """
        custom_emoji_regex = re.compile("["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002500-\U00002BEF"
                                        u"\U00002702-\U000027B0"
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001f926-\U0001f937"
                                        u"\U00010000-\U0010ffff"
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u200d"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\ufe0f"  # dingbats
                                        u"\u3030"
                                        "]+", re.UNICODE)
        return re.sub(custom_emoji_regex, '', data)

    @staticmethod
    def punctuation_remover(text: str) -> str:
        """
        Remove punctuation from the input text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with punctuation removed.
        """
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub(" ", text)
        return text


class DataPreparing:
    """
    Class for preparing data_handlers for sentiment analysis.
    """

    TEXT_COLUMN = 'Text'
    SCORE_COLUMN = 'Score'

    def __init__(self, file_path: str, config: CustomConfig) -> None:
        """
        Initializes the DataPreparing class.

        Args:
            file_path (str): Path to the CSV file.
            config (CustomConfig): Configuration class for model training.
        """
        self.file_path = file_path
        self.config = config
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    @staticmethod
    def helper_label_mapping(x: int) -> int:
        """
        Map numerical labels to sentiment classes.

        Args:
            x (int): Numerical label.

        Returns:
            int: Mapped sentiment class.
        """
        if x in [1, 2]:
            return 0  # Negative
        elif x == 3:
            return 1  # Neutral
        elif x in [4, 5]:
            return 2  # Positive

    @staticmethod
    def helper_label_naming(x: int) -> str:
        """
        Map numerical labels to sentiment class names.

        Args:
            x (int): Numerical label.

        Returns:
            str: Mapped sentiment class name.
        """
        if x == 0:
            return "Negative"
        elif x == 1:
            return "Neutral"
        elif x == 2:
            return "Positive"

    def read_csv(self):
        """
        Read the CSV file.

        Raises:
            FileNotFoundError: If the CSV file is not found.
        """
        try:
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {self.file_path}")

    def map_labels(self):
        """Map numerical labels to sentiment classes and names."""
        self.df["label"] = self.df[self.SCORE_COLUMN].apply(lambda x: self.helper_label_mapping(x))
        self.df["label_name"] = self.df["label"].apply(lambda x: self.helper_label_naming(x))

    def _remove_emojis(self):
        """Remove emojis from the 'text' column."""
        self.df["text"] = self.df[self.TEXT_COLUMN].apply(lambda x: TextProcessing.emoji_remover(x))

    def _remove_punctuation(self):
        """Remove punctuation from the 'text' column."""
        cleaner = CustomTextCleaner()
        self.df["text"] = self.df["text"].apply(lambda x: TextProcessing.punctuation_remover(x))

    def _lowercase_text(self):
        """Convert the 'text' column to lowercase."""
        self.df["text"] = self.df["text"].apply(lambda x: x.lower())

    def _apply_cleaner(self):
        """Apply the CustomTextCleaner to the 'text' column."""
        cleaner = CustomTextCleaner()
        self.df["text"] = self.df[self.TEXT_COLUMN].apply(lambda x: cleaner(x)[0][0])

    def clean_text(self):
        """Clean the text in the DataFrame."""
        self._remove_emojis()
        self._lowercase_text()
        self._apply_cleaner()
        self._remove_punctuation()

    def remove_unwanted_columns(self):
        """Remove unwanted columns from the DataFrame."""
        columns_to_keep = ['text', 'label', 'label_name']
        self.df = self.df[columns_to_keep]

    def resample_classes(self):
        """Resample minority classes to balance the dataset."""
        df_majority_positive = self.df[self.df['label'] == 2]  # positive class
        df_minority_neg = self.df[self.df['label'] == 0]  # negative class
        df_minority_neutral = self.df[self.df['label'] == 1]  # neutral class

        df_minority_neg_upsampled = resample(df_minority_neg, replace=True,
                                             n_samples=self.config.desired_samples,
                                             random_state=self.config.random_state)
        df_minority_neutral_upsampled = resample(df_minority_neutral, replace=True,
                                                 n_samples=self.config.desired_samples,
                                                 random_state=self.config.random_state)
        df_majority_positive_upsampled = resample(df_majority_positive, replace=True,
                                                  n_samples=self.config.desired_samples,
                                                  random_state=self.config.random_state)

        self.df = pd.concat([df_majority_positive_upsampled, df_minority_neg_upsampled, df_minority_neutral_upsampled])
        self.df = self.df.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)

    def split_train_val_test(self):
        """Split the dataset into training, validation, and test sets."""
        # Split the dataset into train and temporary (val + test) parts
        train_part, temp_part = train_test_split(
            self.df, test_size=(self.config.val_size + self.config.test_size),
            random_state=self.config.random_state,
            stratify=self.df['label'].values)

        # Split the temporary part into val and test parts
        self.val_df, self.test_df = train_test_split(
            temp_part,
            test_size=self.config.test_size / (self.config.val_size + self.config.test_size),
            random_state=self.config.random_state,
            stratify=temp_part['label'].values)

        # Assign the train part to the train_df
        self.train_df = train_part

    def prepare_data(self):
        """Prepare the data_handlers for sentiment analysis."""
        self.read_csv()
        self.map_labels()
        self.clean_text()
        self.remove_unwanted_columns()
        self.resample_classes()
        self.split_train_val_test()


class CustomTokenizer:
    """Class for tokenizing reviews"""

    def __init__(self, train_df, val_df, test_df, config: CustomConfig, text_column='text', label_column='label'):
        """
        Initializes the Tokenizer class.

        Args:
            train_df (pd.DataFrame): Training dataset.
            val_df (pd.DataFrame): Validation dataset.
            test_df (pd.DataFrame): Test dataset.
            config (CustomConfig): Configuration class for model training.
            text_column (str): Name of the text column.
            label_column (str): Name of the label column.
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.config = config
        self.tokenizer = None
        self.text_column = text_column
        self.label_column = label_column
        self.datasets = {}

    def initialize_bert_tokenizer(self, do_lower_case=True):
        """Initialize the BERT tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(self.config.pretrained_model, do_lower_case=do_lower_case)

    def _tokenize_dataset(self, df: pd.DataFrame, key: str) -> None:
        """
        Tokenize a dataset.

        Args:
            df (pd.DataFrame): DataFrame containing text data_handlers.
            key (str): Key to identify the dataset.

        Returns:
            None
        """
        data = df[self.text_column].values
        encoded_data = self.tokenizer.batch_encode_plus(
            data,
            add_special_tokens=self.config.add_special_tokens,
            return_attention_mask=self.config.return_attention_mask,
            pad_to_max_length=self.config.pad_to_max_length,
            max_length=self.config.max_length,
            return_tensors=self.config.return_tensors
        )

        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        labels = torch.tensor(df[self.label_column].values)

        self.datasets[key] = TensorDataset(input_ids, attention_masks, labels)

    def run_tokenizer(self) -> None:
        """Run the BERT tokenizer on training, validation, and test datasets."""
        self.initialize_bert_tokenizer()
        self._tokenize_dataset(self.train_df, key='train')
        self._tokenize_dataset(self.val_df, key='val')
        self._tokenize_dataset(self.test_df, key='test')

    def save_tokenizer(self, save_dir: str) -> None:
        """
        Save the tokenizer to a local directory.

        Args:
            save_dir (str): Directory to save the tokenizer.

        Returns:
            None
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        tokenizer_path = os.path.join(save_dir, 'bert_tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
