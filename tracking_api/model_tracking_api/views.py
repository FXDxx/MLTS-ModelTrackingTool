from rest_framework.viewsets import ModelViewSet
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import MLModels
from .serializers import MLModelsSerializer
from .ml_registry import MODEL_PARAMETER_SCHEMA
# Create your views here.

class MLModelViewSet(ModelViewSet):
    queryset = MLModels.objects.all()
    serializer_class = MLModelsSerializer

    @action(detail=False, methods=['get'], url_path='parameters/(?P<model_type>[^/.]+)')
    def get_parameters(self, request, model_type=None):
        params = MODEL_PARAMETER_SCHEMA.get(model_type)

        if not params:
            return Response({"error": "Invalid model type"}, status=400)

        return Response({"parameters": list(params.keys())})
