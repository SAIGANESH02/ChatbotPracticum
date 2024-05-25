import torch
from transformers import BertTokenizer, BertModel

def load_model(model_path, tokenizer_path):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
        # assuming want the mean of the last hidden states
        features = outputs.last_hidden_state.mean(dim=1)
    return features

# example usage
tokenizer, model = load_model('model_directory_path', 'tokenizer_directory_path')
text = "Example text for prediction."
prediction = predict(text, tokenizer, model)
print(prediction)
