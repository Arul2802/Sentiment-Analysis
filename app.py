from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = load_model('model_name.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None

# Load tokenizer
try:
    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading tokenizer: {e}")
    tokenizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not tokenizer:
        return jsonify({'error': 'Model or tokenizer not loaded. Please check files.'})

    review = request.form.get('review')
    category = request.form.get('category')  # Either "movie" or "book"

    if not review:
        return jsonify({'error': 'No review provided'})

    if category not in ['movie', 'book']:
        return jsonify({'error': 'Invalid category selected'})

    # Tokenize and pad the input review
    try:
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=200)  # Adjust maxlen as per your model's input size
        prediction = model.predict(padded_sequence)
        sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
        confidence = round(float(prediction[0][0]) * 100, 2)
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

    return jsonify({'sentiment': sentiment, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
