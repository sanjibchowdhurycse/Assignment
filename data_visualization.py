import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()

# Plot first 12 digit images
plt.figure(figsize=(8, 6))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Label: {digits.target[i]}")
    plt.axis("off")

plt.suptitle("Sample Digits from Dataset", fontsize=16)
plt.tight_layout()
plt.savefig("sample_digits.png")
plt.show()
