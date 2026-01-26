from django.db import models
from django.core.validators import FileExtensionValidator
# Create your models here.
class MLModels(models.Model):
    name = models.CharField(max_length=255)
    framework = models.CharField(max_length=50)
    model_file = models.FileField(upload_to='models/', validators=[FileExtensionValidator(allowed_extensions=['csv', 'json', 'xlsx', 'tsv', 'parquet', 'mp3', 'wav','flac','jpg','png','tiff','xml'])])
    parameters = models.JSONField(default=dict,null=True, blank=True)
    version = models.IntegerField(default=1)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.name
