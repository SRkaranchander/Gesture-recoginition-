import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ----------------------------------------------------------------------
# 1. Gesture class labels (10 classes)
# ----------------------------------------------------------------------
gesture_labels = [
    "Start", "Stop", "Turn Left", "Turn Right", "Upward",
    "Downward", "Turn Up", "Turn Down", "Grab", "Release"
]

# ----------------------------------------------------------------------
# 2. Ground-truth and predicted labels for the 100-sample test set
#    (10 samples per class, reconstructed to match the reported matrix)
# ----------------------------------------------------------------------
y_true = []
y_pred = []

# Start: 10/10 correct
y_true += [0]*10; y_pred += [0]*10

# Stop: 8 correct, 2 misclassified as Release (9)
y_true += [1]*10; y_pred += [1]*8 + [9]*2

# Turn Left: 10/10 correct
y_true += [2]*10; y_pred += [2]*10

# Turn Right: 10/10 correct
y_true += [3]*10; y_pred += [3]*10

# Upward: 10/10 correct
y_true += [4]*10; y_pred += [4]*10

# Downward: 10/10 correct
y_true += [5]*10; y_pred += [5]*10

# Turn Up: 10/10 correct
y_true += [6]*10; y_pred += [6]*10

# Turn Down: 10/10 correct
y_true += [7]*10; y_pred += [7]*10

# Grab: 10/10 correct
y_true += [8]*10; y_pred += [8]*10

# Release: 8 correct, 2 misclassified as Grab (8)
y_true += [9]*10; y_pred += [9]*8 + [8]*2

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ----------------------------------------------------------------------
# 3. Compute confusion matrix
# ----------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=range(10))

print("Confusion Matrix (N = 100 Test Samples)")
print(cm)

# ----------------------------------------------------------------------
# 4. Per-class precision / recall / F1
# ----------------------------------------------------------------------
report = classification_report(
    y_true, y_pred, target_names=gesture_labels, digits=3
)
print("\nClassification Report")
print(report)

# ----------------------------------------------------------------------
# 5. Plot heatmap
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 7))
short_labels = ["St", "Sp", "TL", "TR", "Up", "Dn", "TU", "TD", "Gr", "Re"]

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=short_labels, yticklabels=short_labels,
    cbar=True, linewidths=0.5, linecolor="gray"
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix — Random Forest Gesture Classifier (N=100)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()