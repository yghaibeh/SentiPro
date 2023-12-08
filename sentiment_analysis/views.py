"""
View for handling chat functionality in the sentiment_analysis_project.

This module defines a Django View class (`ChatView`) responsible for rendering
the chat template and handling POST requests with chat prompts for sentiment analysis.
"""

from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
import json
from django.apps import apps

# Access the SentimentAnalysis AppConfig
sentiment_app_config = apps.get_app_config('sentiment_analysis')

# Access the sentiment_model from the AppConfig
sentiment_model = sentiment_app_config.sentiment_model


class ChatView(View):
    """
    View class for handling chat functionality.

    Attributes:
        template_name (str): The template file name for rendering the chat interface.
    """
    template_name = 'sentiment_analysis/chat.html'

    def get(self, request, *args, **kwargs):
        """
        Handle GET requests.

        Args:
            request (HttpRequest): The HTTP GET request object.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            HttpResponse: The rendered chat template.
        """
        return render(request, self.template_name)

    def post(self, request, *args, **kwargs):
        """
        Handle POST requests with chat prompts.

        Args:
            request (HttpRequest): The HTTP POST request object containing chat prompt.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            JsonResponse: JSON response with sentiment analysis results.
        """
        try:
            # Decode JSON data from the request body
            data = json.loads(request.body.decode('utf-8'))
            prompt = data.get('prompt', '')
            results = sentiment_model.infer_with_percentages(prompt)

            response = {
                "label": results["predicted_label"],
                "positive": results['probabilities'][2],
                "neutral": results['probabilities'][1],
                "negative": results['probabilities'][0],
            }

            return JsonResponse(response)
        except Exception as e:
            print('Error:', str(e))
            return JsonResponse({"error": f"Error: {str(e)}"}, status=400)
