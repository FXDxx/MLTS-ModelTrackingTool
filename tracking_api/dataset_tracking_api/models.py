from django.db import models
from django.core.validators import FileExtensionValidator
# Create your models here.
class Datasets(models.Model):
    try:
        name = models.CharField(max_length=200)
        document = models.FileField(upload_to='datasets/', 
                                validators=[FileExtensionValidator(allowed_extensions=['csv', 'json', 'xlsx', 'tsv', 'parquet', 'mp3', 'wav','flac','jpg','png','tiff','xml'])])
        hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
        version = models.IntegerField(default=1)
        created_at = models.DateTimeField(auto_now_add=True)
    except Exception as e:
        print(f"Error in defining Datasets model: {e}")

        def __str__(self):
            return f"{self.name} (v{self.version})"