from rest_framework import serializers
from .models import ModelInference


class ModelInferenceSerializer(serializers.ModelSerializer):
    class Meta:
        try:
            model = ModelInference
            fields = '__all__'
            read_only_fields = ['model','dataset','parameter_used','metrics','status','created_at']
        except Exception as e:
            print(f"Error initializing ModelInferenceSerializer Meta: {e}")