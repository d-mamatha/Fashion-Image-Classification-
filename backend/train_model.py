import os
import pickle
import numpy as np
from features import extract_features
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "..", "dataset", "train")

print("📁 TRAIN_DIR:", TRAIN_DIR)

# ---------------------------
# Load dataset
# ---------------------------
X = []
y = []

labels = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])

print("📌 Classes:", labels)

for label in labels:
    folder = os.path.join(TRAIN_DIR, label)
    print(f"📂 Loading class: {label}")

    for img_name in os.listdir(folder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder, img_name)

        try:
            feat = extract_features(img_path)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"❌ Skipped {img_path}: {e}")

# ---------------------------
# Convert to numpy
# ---------------------------
X = np.array(X)
y = np.array(y)

print("✅ Feature shape:", X.shape)
print("✅ Labels count:", len(y))

if len(X) == 0:
    raise RuntimeError("❌ No training images loaded. Check dataset/train folder.")

# ---------------------------
# Encode labels
# ---------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------------------
# ELM Training
# ---------------------------
input_dim = X.shape[1]
hidden_dim = 1500
num_classes = len(labels)

W = np.random.randn(input_dim, hidden_dim)
b = np.random.randn(hidden_dim)

H = np.tanh(np.dot(X, W) + b)
Y_onehot = np.eye(num_classes)[y_encoded]

beta = np.linalg.pinv(H) @ Y_onehot

# ---------------------------
# Training Accuracy
# ---------------------------
preds = np.argmax(H @ beta, axis=1)
train_acc = accuracy_score(y_encoded, preds) * 100

print(f"🎯 Training Accuracy: {train_acc:.2f}%")

# ---------------------------
# Save model
# ---------------------------
with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump((W, b, beta, le.classes_), f)

print("✅ Model trained and saved as model.pkl")
