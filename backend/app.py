from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from model import predict_diabetes  # This loads the model, scaler, selected_indices

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert features list to numpy array
        features = np.array(data['features'], dtype=float)  # raw 1D array
        prediction = predict_diabetes(features)  # No reshape needed; it's handled inside

        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
