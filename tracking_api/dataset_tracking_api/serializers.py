from rest_framework import serializers
from .models import Datasets
from django.utils import timezone
class DatasetsSerializer(serializers.ModelSerializer):
    hash = serializers.CharField(read_only=True)
    version = serializers.IntegerField(read_only=True)
    created_at = serializers.DateTimeField(read_only=True)

    class Meta:
            model = Datasets
            fields = ['id', 'name', 'document', 'hash', 'version', 'created_at']
    def validate_duration(self, value):
        if value <= 0:
            raise serializers.ValidationError("Duration must be a positive integer.")
        return value

    def validate(self, data):
         if data['date'] > timezone.now().date():
             raise serializers.ValidationError("Date cannot be in the future.")
         return data

        

