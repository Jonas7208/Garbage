
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def classification_metrics():
    # Keep the first two metrics stable for existing summary consumers.
    return [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
        tf.keras.metrics.F1Score(average="macro", threshold=None, name="macro_f1"),
    ]


def report_from_predictions(y_true, y_pred, class_names):
    """Compute argmax metrics over the entire dataset, not batch averages."""
    size = len(class_names)
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = matrix.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(size), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(size), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(size), where=(precision + recall) != 0)
    return {
        "split": "validation",
        "class_names": list(class_names),
        "confusion_matrix": matrix.tolist(),
        "matrix_axes": {"rows": "true_class", "columns": "predicted_class"},
        "accuracy": float(tp.sum() / matrix.sum()) if matrix.sum() else 0.0,
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()) if np.any(support) else 0.0,
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
    }


def save_validation_report(model, dataset, class_names, output_path):
    true, predicted = [], []
    # DirectoryIterator repeats indefinitely; access exactly len(dataset) batches.
    batches = dataset if isinstance(dataset, tf.data.Dataset) else (
        dataset[i] for i in range(len(dataset)))
    for images, labels in batches:
        probabilities = np.asarray(model(images, training=False))
        true.extend(np.argmax(np.asarray(labels), axis=1).tolist())
        predicted.extend(np.argmax(probabilities, axis=1).tolist())
    report = report_from_predictions(true, predicted, class_names)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nValidierung: Precision / Recall / F1 / Anzahl")
    for name, row in report["per_class"].items():
        print(f"  {name}: {row['precision']:.3f} / {row['recall']:.3f} / "
              f"{row['f1']:.3f} / {row['support']}")
    print("Verwechslungsmatrix (Zeilen: tatsächlich, Spalten: vorhergesagt):")
    print(np.asarray(report["confusion_matrix"]))
    print(f"Macro-F1: {report['macro_f1']:.4f}; "
          f"Balanced Accuracy: {report['balanced_accuracy']:.4f}")
    print(f"Bericht gespeichert: {path}")
    return report
