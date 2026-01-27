from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet
from .models import ModelInference
from .serializers import ModelInferenceSerializer

# Create your views here.
class ModelInferenceViewset(ModelViewSet):
    try:
        queryset = ModelInference.objects.all()
        serializer_class = ModelInferenceSerializer
    except Exception as e:
        print(f"Error initializing ModelInferenceViewset: {e}")