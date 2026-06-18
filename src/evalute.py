from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from predict import y_pred, y_true

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
plt.imshow(cm)
plt.colorbar()

plt.xticks(
    range(5),
    [
        "Neo",
        "Dig",
        "Nerv",
        "Card",
        "Gen"
    ],
    rotation=45
)

plt.yticks(
    range(5),
    [
        "Neo",
        "Dig",
        "Nerv",
        "Card",
        "Gen"
    ]
)

plt.tight_layout()

plt.savefig("results/confusion_matrix.png")