# Emotion Classification for Chatbot

## Overview
This project implements a suicidal risk classification model using BERT/Logistic Regression to enhance the capabilities of a chatbot by accurately identifying and responding to user emotions. The model is trained to detect various three states(Positive, Negative and Neutral) from text inputs, allowing the chatbot to tailor responses and improve user interactions.

## Repository Structure

- **app.py**: Entry point for the chatbot application integrating the emotion classifier.
- **chatbot.py**, **chatbot_app.py**, **chatbox_v2.py**: Core chatbot scripts illustrating different versions or components of the chatbot interface.
- **data_prep.py**: Script for preparing and preprocessing the data for model training.
- **SameSameClassification.ipynb**, **UpdatedSameSameClassification.ipynb**: Jupyter notebooks containing the model training and evaluation process.
- **model_loader.py**: Helper script for loading the trained BERT model.
- **logistic_regression_model.pkl**: An alternative model for emotion classification.
- **requirements.txt**: Required libraries and dependencies for running the project.

## Installation

To set up the project environment:

1. Clone the repository:
   ```bash
   git clone https://your-repository-link.git
   cd your-repository-directory
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the chatbot with emotion classification:

```bash
python app.py
```

For training the model with new data or re-training:

```bash
jupyter notebook SameSameClassification.ipynb
```

Follow the instructions in the notebook to perform model training and evaluation.
