import pandas as pd
import re
import emoji

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Convert emojis to words
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Remove special characters except emoticons
    text = re.sub(r'[^\w\s,]', '', text)
    return text

# Load data
data = pd.read_excel(r'messages_20240520.xlsx')

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])
data = data.dropna(subset=['Text'])  # Drop rows with missing text
data['Text'] = data['Text'].astype(str).apply(preprocess_text)  # Ensure text column is string type

# Save preprocessed data
data.to_csv('preprocessed_data_cleaned.csv', index=False)
