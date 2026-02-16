import os
import cv2
import pickle
import numpy as np
from features import extract_features
from elm import ELM

# Path to different dataset
DIFF_DIR = r"..\dataset\different"

# Load trained model
with open("model.pkl", "rb") as f:
    W, b, beta, labels = pickle.load(f)

model = ELM(W=W, b=b, beta=beta)

X_diff = []
y_diff = []

# Load images from different dataset
for idx, label in enumerate(labels):
    folder = os.path.join(DIFF_DIR, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path)
            feat = extract_features(img)
            X_diff.append(feat)
            y_diff.append(idx)
        except:
            pass

X_diff = np.array(X_diff)
y_diff = np.array(y_diff)

# Predict
predictions = model.predict(X_diff)
predicted_labels = np.argmax(predictions, axis=1)

# Accuracy
different_accuracy = np.mean(predicted_labels == y_diff) * 100

print(f"🎯 Accuracy on Different Dataset: {different_accuracy:.2f}%")
print(f"Total samples tested: {len(y_diff)}")
