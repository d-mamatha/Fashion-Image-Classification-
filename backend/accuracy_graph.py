import matplotlib.pyplot as plt

labels = [
    "Training Accuracy",
    "Testing Accuracy",
    "Different Dataset Accuracy"
]

accuracy = [100, 88, 76.28]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, accuracy)

plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison of Clothing Classification Model")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 5,
        f"{bar.get_height():.2f}%",
        ha="center",
        color="white",
        fontsize=11
    )

plt.savefig("accuracy_comparison.png")
plt.show()
