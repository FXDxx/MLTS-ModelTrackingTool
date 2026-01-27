import os
import json
from django.core.files import File
from django.conf import settings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    }

def save_predictions_file(run_id, predictions):
    filename = f"inference_{run_id}.json"
    filepath = os.path.join(settings.MEDIA_ROOT, "inference_outputs", filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save predictions as JSON
    # If predictions is a NumPy array, convert to list first
    if not isinstance(predictions, list):
        predictions = predictions.tolist()

    with open(filepath, "w") as f:
        json.dump({"predictions": predictions}, f, indent=4)

    # Return relative path for FileField
    return f"inference_outputs/{filename}"
