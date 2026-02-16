import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2

# Load CNN ONCE
cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

def extract_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not readable")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = cnn.predict(img, verbose=0)
    return features.flatten()
