import matplotlib.pyplot as plt

labels = ["Training Accuracy", "Testing Accuracy", "Different Dataset Accuracy"]
accuracies = [100, 98, 92]  # update if needed

plt.figure(figsize=(7,5))
plt.bar(labels, accuracies)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.title("Accuracy Comparison of CNN + ELM Model")

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc}%", ha='center')

plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()
