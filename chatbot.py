# chatbot_app.py
import openai
import streamlit as st
import pandas as pd

# Load your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Define functions to interact with GPT-3.5
def create_classification_prompt(text):
    return f"Classify the mental health concern in the following message: {text}"

def create_sentiment_prompt(text):
    return f"Analyze the sentiment of the following message: {text}"

def gpt3_completion(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",  # Use GPT-3.5 model
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def classify_message(message):
    prompt = create_classification_prompt(message)
    classification = gpt3_completion(prompt)
    return classification

def analyze_sentiment(message):
    prompt = create_sentiment_prompt(message)
    sentiment = gpt3_completion(prompt)
    return sentiment

def automated_response(classification, sentiment):
    if 'acute' in classification.lower():
        return "Alerting moderator: Immediate attention needed!"
    elif 'positive' in sentiment.lower():
        return "Thank you for your positive message!"
    else:
        return "We are here to support you. Please reach out if you need help."

def process_message(message):
    classification = classify_message(message)
    sentiment = analyze_sentiment(message)
    response = automated_response(classification, sentiment)
    return classification, sentiment, response

# Streamlit app layout
st.title("Mental Health Chatbot")
st.write("This chatbot helps to classify mental health concerns and analyze sentiment in messages.")

# Chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# User input
user_input = st.text_area("Enter your message:", "", key="input_area")

if st.button("Send"):
    if user_input:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(process_message, user_input)
            classification, sentiment, score, response = future.result()

        # Append user query and bot response to history
        st.session_state['history'].append({"user": user_input, "bot": response})

        # Clear input
        st.session_state['input_area'] = ""

# Display chat history
for chat in st.session_state['history']:
    st.chat_message("user").markdown(chat['user'])
    st.chat_message("bot").markdown(chat['bot'])
