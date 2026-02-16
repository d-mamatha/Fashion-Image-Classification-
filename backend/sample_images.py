import os
import matplotlib.pyplot as plt
from PIL import Image
import random

DATASET_DIR = "../dataset/train"
classes = os.listdir(DATASET_DIR)

plt.figure(figsize=(12, 8))

for i, cls in enumerate(classes):
    img_name = random.choice(os.listdir(os.path.join(DATASET_DIR, cls)))
    img_path = os.path.join(DATASET_DIR, cls, img_name)

    img = Image.open(img_path)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_dataset_images.png")
plt.show()
