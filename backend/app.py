from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import pickle
import tempfile
import os

from features import extract_features

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load trained ELM model
# --------------------------------------------------
with open("model.pkl", "rb") as f:
    W, b, beta, labels = pickle.load(f)

# --------------------------------------------------
# ELM forward pass (NO class needed)
# --------------------------------------------------
def elm_predict(X):
    H = np.tanh(np.dot(X, W) + b)
    output = np.dot(H, beta)
    return np.argmax(output, axis=1)

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()

    # Save temporarily (because extract_features expects path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        # Extract CNN features
        features = extract_features(tmp_path).reshape(1, -1)

        # Predict using ELM
        pred_idx = elm_predict(features)[0]

        return {"prediction": labels[pred_idx]}

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
