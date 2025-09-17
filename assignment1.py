import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

# Calculate metrics
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

acc_nb = accuracy_score(y_test, y_pred_nb)
acc_lr = accuracy_score(y_test, y_pred_lr)

report_nb = classification_report(y_test, y_pred_nb, output_dict=True)
report_lr = classification_report(y_test, y_pred_lr, output_dict=True)

precision_nb = report_nb["weighted avg"]["precision"]
precision_lr = report_lr["weighted avg"]["precision"]

recall_nb = report_nb["weighted avg"]["recall"]
recall_lr = report_lr["weighted avg"]["recall"]

f1_nb = report_nb["weighted avg"]["f1-score"]
f1_lr = report_lr["weighted avg"]["f1-score"]

nb_scores = [acc_nb, precision_nb, recall_nb, f1_nb]
lr_scores = [acc_lr, precision_lr, recall_lr, f1_lr]

# Plot all metrics in one figure
import numpy as np

x = np.arange(len(metrics))  # metric indices
bar_width = 0.35

plt.figure(figsize=(10,6))
plt.bar(x - bar_width/2, nb_scores, width=bar_width, label="Naive Bayes", color="skyblue")
plt.bar(x + bar_width/2, lr_scores, width=bar_width, label="Logistic Regression", color="lightgreen")

plt.xticks(x, metrics)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Comparison of Naive Bayes and Logistic Regression")
plt.legend()
plt.savefig("model_comparison.png")
plt.show()
