from django.db import models
from model_tracking_api.models import MLModels as MLModel
from dataset_tracking_api.models import Datasets
# Create your models here.

class ModelInference(models.Model):
    try:
        model = models.ForeignKey(MLModel, on_delete=models.CASCADE) 
        dataset = models.ForeignKey(Datasets, on_delete=models.CASCADE)
        parameter_used = models.JSONField(default=dict, blank=True)
        metrics = models.JSONField(default=dict, blank=True)
        status = models.CharField(max_length=50, default='pending')
        created_at = models.DateTimeField(auto_now_add=True)
    except Exception as e:
        print(f"Error initializing ModelInference model: {e}")