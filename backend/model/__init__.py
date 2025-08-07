import joblib
import numpy as np

model = joblib.load('model/svm_model.pkl')
selected_indices = joblib.load('model/selected_indices.pkl')
scaler = joblib.load('model/scaler.pkl')

def predict_diabetes(features):
    scaled = scaler.transform([features])  # scale single input
    selected = scaled[:, selected_indices]  # select optimized features
    prediction = model.predict(selected)
    return prediction[0]
