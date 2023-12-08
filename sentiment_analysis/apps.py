from django.apps import AppConfig
from django.conf import settings


class SentimentAnalysis(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sentiment_analysis'

    def __init__(self, app_name, app_module):
        super().__init__(app_name, app_module)
        self.sentiment_model = None

    def ready(self):
        pass
        # Import the SentimentAnalysisInference class and initialize it here
        from model_handler.model_inference import SentimentAnalysisInference
        from utils.config_utils import CustomConfig

        # Initialize SentimentAnalysisInference only once when the project runs
        self.sentiment_model = SentimentAnalysisInference(settings.SENTIMENT_MODEL_PATH, settings.SENTIMENT_CONFIG)
