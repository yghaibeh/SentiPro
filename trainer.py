import torch
import logging
import numpy as np
import random

from transformers import BertForSequenceClassification

from data_handlers.data_processing import DataPreparing, CustomTokenizer
from model_handler.data_loader import CustomDataLoader
from model_handler.model_trainer import SentimentAnalysisModel
from utils.config_utils import CustomConfig
from utils.visualizations import Visualizer

# Load the configuration
config = CustomConfig()

# Set seed for reproducibility
random.seed(config.seed_val)
np.random.seed(config.seed_val)
torch.manual_seed(config.seed_val)
torch.cuda.manual_seed_all(config.seed_val)


def setup_logger():
    """
    Set up logging configuration.
    """
    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)


if __name__ == "__main__":
    # Set up logger
    setup_logger()

    # Log the device being used (CPU or GPU)
    logging.info(f"Using device: {config.device}")

    # Initialize DataPreparing class to prepare data_handlers for sentiment analysis
    logging.info("Initializing DataPreparing to prepare data_handlers for sentiment analysis.")
    data_preparer = DataPreparing(file_path="datasets/amazon_dataset/Reviews.csv", config=config)
    data_preparer.prepare_data()

    # Visualize the count of labels in the dataset
    logging.info("Visualizing the count of labels in the dataset.")
    # Visualizer.show_count_of_labels(data_preparer.df, 'label')

    # Split the dataset into training, validation, and test sets
    logging.info("Splitting the dataset into training, validation, and test sets.")
    train_df = data_preparer.train_df
    val_df = data_preparer.val_df
    test_df = data_preparer.test_df

    # Initialize CustomTokenizer for tokenizing reviews
    logging.info("Initializing CustomTokenizer for tokenizing reviews.")
    tokenizer = CustomTokenizer(train_df=train_df, val_df=val_df, test_df=test_df, config=config)
    tokenizer.run_tokenizer()

    # Get tokenized datasets for training and validation
    logging.info("Getting tokenized datasets for training and validation.")
    dataset_train = tokenizer.datasets['train']
    dataset_val = tokenizer.datasets['val']
    dataset_test = tokenizer.datasets['test']

    # Create custom data_handlers loaders for training and validation
    logging.info("Creating custom data_handlers loaders for training and validation.")
    data_loader = CustomDataLoader(dataset_train, dataset_val, dataset_test, config)

    # Initialize SentimentAnalysisModel for sentiment analysis
    logging.info("Initializing SentimentAnalysisModel for sentiment analysis.")
    sentiment_model = SentimentAnalysisModel(data_loader.dataloader_train,
                                             data_loader.dataloader_val,
                                             data_loader.dataloader_test,
                                             config)

    logging.info("Setting up the model architecture.")
    sentiment_model.set_model()

    # Set the device for training (CPU or GPU)
    logging.info(f"Setting up the device for training: {config.device}.")
    sentiment_model.set_device()

    # Set up the optimizer for model training
    logging.info("Setting up the optimizer for model training.")
    sentiment_model.set_optimizer()

    # Set up the learning rate scheduler
    logging.info("Setting up the learning rate scheduler.")
    sentiment_model.set_scheduler()

    # Train the sentiment analysis model
    logging.info("Training the sentiment analysis model.")
    # sentiment_model.train()

    # Train the sentiment analysis model
    logging.info("Testing the sentiment analysis model.")
    # sentiment_model.train()

    # Test the sentiment analysis model
    label_dict = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    model_path = 'saved_models/_BERT_epoch_3.model'
    sentiment_model.load_and_evaluate_on_saved_model(model_path=model_path, label_dict=label_dict)
