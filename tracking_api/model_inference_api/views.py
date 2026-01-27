from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet
from rest_framework.decorators import action
from rest_framework.response import Response
from .serializers import ModelInferenceSerializer

# Create your views here.
class ModelInferenceViewset(ModelViewSet):
    queryset = ModelInferenceSerializer.objects.all()
    serializer_class = ModelInferenceSerializer
