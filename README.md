# Emotion Classification for Chatbot

## Overview
This project implements a suicidal risk classification model using BERT/Logistic Regression to enhance the capabilities of a chatbot by accurately identifying and responding to user emotions. The model is trained to detect three states(Positive, Negative and Neutral) from text inputs, allowing the chatbot to tailor responses and improve user interactions.

## Repository Structure

- **app.py**: Entry point for the chatbot application integrating the emotion classifier with Logistic Regression/BERT.
- **chatbot.py**: Chatbot application integrating the emotion classifier with OpenAI GPT-3.5 model.
- **chatbot_app.py**: Chatbot application integrating the emotion classifier with OpenAI GPTJ model.
- **chatbox_v2.py**: Chatbot scripts illustrating model efficiency.
- **data_prep.py**: Script for preparing and preprocessing the data for model training.
- **SameSameClassification.ipynb**, **UpdatedSameSameClassification.ipynb**: Jupyter notebooks containing the Logistic Regression and BERT model training and evaluation processes.
- **model_loader.py**: Helper script for loading the trained Logistic Regression/BERT models.
- **logistic_regression_model.pkl**: Specific trained Logistic Regression model parameters.
- **practicum_cb (1).ipynb**: Jupyter notebooks containing the prompt engineering and alerting mechanism.
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
