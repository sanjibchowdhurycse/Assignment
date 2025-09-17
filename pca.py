from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(digits.data)

# Plot PCA 2D projection
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                      c=digits.target, cmap="tab10", alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title("PCA Projection of Digits Dataset (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig("pca_digits.png")
plt.show()
