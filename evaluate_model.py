import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import preprocess_emnist

MODEL_PATH = "models/handwriting_cnn.keras"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run: python train_model.py"
    )

model = tf.keras.models.load_model(MODEL_PATH)

test_ds, ds_info = tfds.load(
    "emnist/balanced",
    split="test",
    as_supervised=True,
    with_info=True,
)

label_names = list(ds_info.features["label"].names)

test_ds = test_ds.map(
    preprocess_emnist, num_parallel_calls=tf.data.AUTOTUNE
)
test_ds = test_ds.batch(128).prefetch(tf.data.AUTOTUNE)

loss, accuracy = model.evaluate(test_ds, verbose=1)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

y_true = []
y_pred = []

for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predictions)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
    )
)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(15, 13))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names,
)
plt.title("EMNIST Balanced - CNN Confusion Matrix")
plt.xlabel("Predicted Character")
plt.ylabel("Actual Character")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300,
)
plt.close()

print("\nConfusion matrix saved to outputs/confusion_matrix.png")
