import os
import cv2
import pickle
import numpy as np
from features import extract_features
from elm import ELM

# Path to test dataset
TEST_DIR = r"..\dataset\test"

# Load trained model
with open("model.pkl", "rb") as f:
    W, b, beta, labels = pickle.load(f)

model = ELM(W=W, b=b, beta=beta)

X_test = []
y_test = []

# Load test images
for idx, label in enumerate(labels):
    folder = os.path.join(TEST_DIR, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path)
            feat = extract_features(img)
            X_test.append(feat)
            y_test.append(idx)
        except:
            pass

X_test = np.array(X_test)
y_test = np.array(y_test)

# Predict
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Accuracy
testing_accuracy = np.mean(predicted_labels == y_test) * 100

print(f"🎯 Testing Accuracy: {testing_accuracy:.2f}%")
print(f"Total test samples: {len(y_test)}")
