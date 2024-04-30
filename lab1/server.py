from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import re

app = Flask(__name__)

# Assuming LSTMClassifier is defined elsewhere in the module or imported
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return torch.sigmoid(x)

# Load model and set to evaluation mode
model_path = './model_state_dict.pth'
vocab_size = 5000
embedding_dim = 400
hidden_dim = 256
output_dim = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the clean_text function (or import it if defined elsewhere)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def text_to_sequences(text, tokenizer, maxlen):
    # Tokenize text by splitting and convert to sequence of integers
    sequence = [tokenizer.get(word, tokenizer.get('<UNK>')) for word in text.split()]
    # Pad sequence if necessary
    padded_sequence = np.pad(sequence, (maxlen - len(sequence), 0), mode='constant', constant_values=0) if len(sequence) < maxlen else sequence[:maxlen]
    return np.array([padded_sequence])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    cleaned_text = clean_text(text)
    
    # Prepare the text for the model
    maxlen = 500  # Define the same maxlen as used during training
    sequences = text_to_sequences(cleaned_text, tokenizer, maxlen)
    input_tensor = torch.tensor(sequences, dtype=torch.long).to(device)
    
    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.item() > 0.5

    response = {'prediction': 'real' if prediction else 'fake'}
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
