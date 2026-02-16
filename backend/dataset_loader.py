import os
import numpy as np
from features import extract_features

def load_data(base_path):
    X, y = [], []
    labels = sorted(os.listdir(base_path))

    for idx, label in enumerate(labels):
        folder = os.path.join(base_path, label)
        for img in os.listdir(folder):
            path = os.path.join(folder, img)
            try:
                X.append(extract_features(path))
                y.append(idx)
            except:
                pass

    return np.array(X), np.array(y), labels
