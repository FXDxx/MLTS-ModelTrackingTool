from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet
from .models import ModelInference
from .serializers import ModelInferenceSerializer

# Create your views here.
class ModelInferenceViewset(ModelViewSet):
    queryset = ModelInference.objects.all()
    serializer_class = ModelInferenceSerializer