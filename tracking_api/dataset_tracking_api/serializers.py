from rest_framework import serializers
from .models import Datasets

class DatasetsSerializer(serializers.ModelSerializer):
    hash = serializers.CharField(read_only=True)
    version = serializers.IntegerField(read_only=True)
    created_at = serializers.DateTimeField(read_only=True)
    class Meta:
        model = Datasets
        fields = ['id', 'name', 'document', 'hash', 'version', 'created_at']

