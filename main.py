from typing import List

from model_handler.model_inference import SentimentAnalysisInference
from utils.config_utils import CustomConfig

# Load the configuration
config = CustomConfig()

if __name__ == '__main__':
    # Example usage for inference
    new_text_data: List[str] = ["I'm so happy",]
    model_path: str = 'saved_models/_BERT_epoch_3.model'

    # Instantiate the inference class
    sentiment_inference: SentimentAnalysisInference = SentimentAnalysisInference(model_path=model_path,
                                                                                 config=config)

    # Perform inference
    results = sentiment_inference.infer_with_percentages(new_text_data[0])
    print(results)
