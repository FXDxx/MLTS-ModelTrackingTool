from django.db import models
from django.core.validators import FileExtensionValidator
# Create your models here.
class MLModels(models.Model):
    model_name = models.CharField(max_length=255)
    model_file = models.FileField(upload_to='models/', validators=[FileExtensionValidator(allowed_extensions=['pkl', 'h5', 'pt', 'joblib', 'pth', 'onnx'])])
    parameters = models.JSONField(default=dict,null=True, blank=True)
    version = models.IntegerField(default=1)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.name
