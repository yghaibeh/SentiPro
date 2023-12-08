from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.config_utils import CustomConfig
from typing import Any


class CustomDataLoader:
    """
    A class for creating custom data_handlers loaders for training and validation datasets.
    """

    def __init__(self, dataset_train: Any, dataset_val: Any, dataset_test: Any, config: CustomConfig) -> None:
        """
        Initializes the CustomDataLoader.

        Args:
            dataset_train: Training dataset.
            dataset_val: Validation dataset.
            dataset_test: Test dataset.
            config: Configuration for model training.
        """
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test

        self.config = config

        self.dataloader_train = None
        self.dataloader_val = None

        # Create data_handlers loaders during initialization
        self._create_data_loaders()

    def _create_data_loaders(self) -> None:
        """
        Creates the training and validation data_handlers loaders.
        """
        # Training DataLoader with RandomSampler for shuffling batches during training
        self.dataloader_train = DataLoader(self.dataset_train,
                                           sampler=RandomSampler(self.dataset_train),
                                           batch_size=self.config.batch_size)

        # Validation DataLoader with SequentialSampler for sequential batching during validation
        self.dataloader_val = DataLoader(self.dataset_val,
                                         sampler=SequentialSampler(self.dataset_val),
                                         batch_size=self.config.batch_size)

        # Test DataLoader with SequentialSampler for sequential batching during validation
        self.dataloader_test = DataLoader(self.dataset_val,
                                          sampler=SequentialSampler(self.dataset_val),
                                          batch_size=self.config.batch_size)
