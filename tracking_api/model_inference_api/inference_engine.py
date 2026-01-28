import joblib
import pandas as pd
from django.utils import timezone
from .calculation_functions import calculate_metrics, save_predictions_file
def execute_inference(run):
    run.status = "RUNNING"
    run.started_at = timezone.now()
    run.save()

    model = joblib.load(run.model.model_file.path)
    data = pd.read_csv(run.dataset.document.path)

    X = data.drop(columns=["target"], errors="ignore")
    y_true = data.get("target")

    predictions = model.predict(X)
    output_path = save_predictions_file(run.id, predictions)

    metrics = calculate_metrics(y_true, predictions) if y_true is not None else {}

    run.metrics = metrics
    run.output_file = output_path
    run.status = "COMPLETED"
   
    run.status = "FAILED"
    run.metrics = {"error": str(e)}
  
    run.completed_at = timezone.now()
    run.save()
