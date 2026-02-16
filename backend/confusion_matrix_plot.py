import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from features import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "..", "dataset", "different")

# Load trained model
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    W, b, beta, labels = pickle.load(f)

def elm_predict(X):
    H = np.tanh(np.dot(X, W) + b)
    output = np.dot(H, beta)
    return np.argmax(output, axis=1)

y_true, y_pred = [], []
label_to_idx = {label: i for i, label in enumerate(labels)}

for label in labels:
    class_dir = os.path.join(TEST_DIR, label)
    for img in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img)
        feat = extract_features(img_path).reshape(1, -1)
        pred = elm_predict(feat)[0]
        y_true.append(label_to_idx[label])
        y_pred.append(pred)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Clothing Image Classification (Different Dataset)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
