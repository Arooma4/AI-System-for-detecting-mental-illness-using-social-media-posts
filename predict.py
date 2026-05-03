import tensorflow as tf
import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = tf.keras.models.load_model(os.path.join(BASE_DIR,"models","cnn_model.h5"))
vectorizer = joblib.load(os.path.join(BASE_DIR,"models","vectorizer.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR,"models","label_encoder.pkl"))

def predict_with_confidence(text):

    features = vectorizer.transform([text]).toarray()

    pred = model.predict(features)[0]

    label = encoder.inverse_transform([np.argmax(pred)])[0]

    confidence = float(np.max(pred))*100

    probabilities = dict(zip(encoder.classes_,pred))

    return label,confidence,probabilities