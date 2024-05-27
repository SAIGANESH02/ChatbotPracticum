import chainlit as cl
import pandas as pd
from transformers import pipeline, BertTokenizer, BertModel
import torch
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
from tqdm import tqdm

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

# Load classification model
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def create_classification_prompt(text):
    return f"Classify the mental health concern in the following message: {text}"

def analyze_sentiment(message):
    result = sentiment_pipeline(message)
    return result[0]['label']

def get_contextual_response(classification, sentiment):
    if classification == 1 and sentiment == 'NEGATIVE':
        return 'It sounds like you might be going through something serious. If you need support now, please go directly to your nearest medical clinic or call this toll-free emergency number 112 right now!'
    elif classification == 0 and sentiment == 'NEGATIVE':
        return "I'm sorry to hear that you're feeling this way. It's important to talk to someone who can provide support. Would you like me to find some resources for you?"
    elif sentiment == 'POSITIVE':
        return "I'm glad to hear that you're feeling positive! If there's anything else you need, feel free to let me know."
    else:
        return "Thank you for sharing. Is there anything specific you'd like to talk more about?"

def process_in_batches(text_list, tokenizer, model, batch_size=16):
    all_features = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="Processing Batches"):
        batch_texts = text_list[i:i + batch_size]
        tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**tokens)
            batch_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_features.append(batch_features)

    return np.concatenate(all_features, axis=0)

# Load logistic regression model
joblin_file = "logistic_regression_model.pkl"
lmodel = joblib.load(joblin_file)

# Define the Chainlit app
@cl.on_message
async def handle_message(message):
    tokenizer, model = load_model()
    text = [message.content]
    print(text)
    features_balanced = process_in_batches(text, tokenizer, model, batch_size=1)
    prediction = lmodel.predict(features_balanced)
    sentiment_label = analyze_sentiment(message.content)
    response = get_contextual_response(prediction[0], sentiment_label)
    print(response)
    
    await cl.Message(content=response).send()

# Run the Chainlit app
if __name__ == "__main__":
    cl.run()
