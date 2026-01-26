from rest_framework import serializers
from .models import MLModels

class MLModelsSerializer(serializers.ModelSerializer):
    uploaded_at = serializers.DateTimeField(read_only=True)
    class Meta:
        model = MLModels
        fields = ['name', 'model_file','parameters','version','uploaded_at']
