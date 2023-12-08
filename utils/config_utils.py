from pathlib import Path
import json
from dataclasses import dataclass, asdict
import torch


@dataclass
class CustomConfig:
    """
    Configuration class for model training.

    Attributes:
        seed_val (int): Seed value for random number generation.
        device (torch.device): Device for training (cuda or cpu).
        epochs (int): Number of training epochs.
        num_classes (int): Number of classes in the classification task.
        desired_samples (int): Number of desired samples for each class.
        batch_size (int): Batch size for training.
        max_length (int): Maximum sequence length.
        learning_rate (float): Learning rate for the optimizer.
        eps (float): Epsilon value for numerical stability.
        pretrained_model (str): Pretrained BERT model name.
        val_size (float): Proportion of the dataset to include in the val split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.
        add_special_tokens (bool): Whether to add special tokens in the input sequences.
        return_attention_mask (bool): Whether to return attention masks for input sequences.
        pad_to_max_length (bool): Whether to pad sequences to the maximum length.
        do_lower_case (bool): Whether to convert input text to lowercase.
        return_tensors (str): Return type for the input tensors ('pt' for PyTorch tensors).

    Methods:
        save_as_json(): Save the configuration to a JSON file.
    """
    seed_val: int = 25
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs: int = 5
    num_classes: int = 3
    desired_samples = 42000
    batch_size: int = 6
    max_length: int = 512
    learning_rate: float = 2e-5
    eps: float = 1e-8
    pretrained_model: str = 'bert-base-uncased'
    val_size: float = 0.10
    test_size: float = 0.10
    random_state: int = 42
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    pad_to_max_length: bool = True
    do_lower_case: bool = False
    return_tensors: str = 'pt'

    def save_as_json(self) -> None:
        """
        Save the configuration as a JSON file.
        """
        with Path('config.json').open("w") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)
