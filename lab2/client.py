import requests
import pandas as pd

# Define the URL of your Flask app
url = 'http://localhost:5000/predict'
data = {'text': 'This is a good day'}
response = requests.post(url, json=data)
prediction = response.json()['prediction']
print(f"Prediction: {prediction}")
df = pd.DataFrame({'Query': [data['text']], 'Prediction': [prediction]})
print(df)

df.to_csv('predictions.csv', index=False)
