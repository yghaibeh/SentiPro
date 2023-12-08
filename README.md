# SentiPro - Sentiment Analyser

## Overview
This project implements sentiment analysis using a BERT-based model within a Django web application. The application allows users to input text prompts and receive sentiment analysis results. The sentiment model is trained using a custom dataset, which is derived from the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) on Kaggle.

## Original Database Information

### Kaggle Dataset Link
[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

### Importance of the Database
The Amazon Fine Food Reviews dataset has been widely used in various research papers. One notable paper that utilized this dataset is:
J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. WWW, 2013.


### Dataset Details
- **Context:**
  This dataset consists of reviews of fine foods from Amazon, spanning a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.

- **Contents:**
  - **Reviews.csv:** Pulled from the corresponding SQLite table named Reviews in `database.sqlite`.
  - **database.sqlite:** Contains the table 'Reviews'.

- **Data Includes:**
  - Reviews from Oct 1999 - Oct 2012
  - 568,454 reviews
  - 256,059 users
  - 74,258 products
  - 260 users with > 50 reviews

## Project Structure

### `data_processing.py`
This module includes classes and functions for processing and preparing the sentiment analysis dataset. It involves text cleaning, label mapping, and dataset balancing.

### `data_loader.py`
The `CustomDataLoader` class creates custom data loaders for training, validation, and test datasets.

### `model_inference.py`
The `SentimentAnalysisInference` class handles model inference. It loads a pre-trained sentiment analysis model and performs inference on new text data.

### `model_trainer.py`
The `SentimentAnalysisModel` class defines and trains the sentiment analysis model. It includes functions for training, evaluation, and testing.

### `views.py`
The `ChatView` class in this Django app handles rendering the chat interface and processing user requests for sentiment analysis.

### `utils/`
The `utils` directory contains utility modules such as configuration management (`config_utils.py`), metrics computation (`metrics.py`), and other helper functions.

## Training Results

### Epoch 1
- **Training loss:** 0.5768399631623179
- **Validation loss:** 0.5245223489105701
- **F1 Score (Weighted):** 0.8161223751254332

### Epoch 2
- **Training loss:** 0.4140596124203876
- **Validation loss:** 0.561298894636333
- **F1 Score (Weighted):** 0.8420671788885337

### Epoch 3
- **Training loss:** 0.28194249292159657
- **Validation loss:** 0.7644271336090751
- **F1 Score (Weighted):** 0.8457100068239863


## Model Training Considerations

The sentiment analysis model was initially designed to be trained on a larger dataset of 124k samples for 5 epochs, with an expected accuracy exceeding 95%. However, due to resource and computation limitations, the model was trained on a subset of 30k samples for only 3 epochs. The achieved results are presented above.

For the complete training process and details, refer to the [Colab notebook](https://colab.research.google.com/drive/198m6I-ah7bSmfYBI6aOzshQedRtovEDv?usp=sharing).


## Usage

### Setup
Install required dependencies: `pip install -r requirements.txt`


### Training the Model
To train the sentiment analysis model, run:
```
python train_model.py
```

### Running the Django Development Server
Start the development server:
```
python manage.py runserver
```
Visit http://localhost:8000 in your browser to access the chat interface.


### Web Interface
The web interface allows users to input text prompts and receive sentiment analysis results in real-time. The chat functionality is handled by the ChatView class.


## Additional Details
### Customization
Feel free to customize the code, such as adjusting hyperparameters, modifying the model architecture, or enhancing the web interface.

### Model Evaluation
After training, the model can be evaluated using the evaluate method in `model_trainer.py`.

### Inference with Saved Model
The `load_and_evaluate_on_saved_model` method in `model_trainer.py` allows loading a saved model for evaluation on the validation and test sets.

### Issues and Notes
Document any known issues, limitations, or additional notes related to the project.

## License
Feel free to further customize or expand the content based on your specific project requirements.

## Project Creator
Muhammad Yaman Ghaibeh
Contact: YGhaibeh@hotmail.com

Feel free to let me know if you have any further requests or modifications!

