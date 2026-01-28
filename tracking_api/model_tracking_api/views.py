from rest_framework.viewsets import ModelViewSet
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import MLModels
from .serializers import MLModelsSerializer
from .ml_registry import MODEL_PARAMETER_SCHEMA
from rest_framework.pagination import PageNumberPagination
# Create your views here.

class SmallPagePagination(PageNumberPagination):
    page_size=5
    page_size_query_param='page_size'
    max_page_size=10

class MLModelViewSet(ModelViewSet):
    queryset = MLModels.objects.all()
    serializer_class = MLModelsSerializer
    lookup_field = 'model_name'
    pagination_class = SmallPagePagination

    @action(detail=False, methods=['get'], url_path='parameters/(?P<model_type>[^/.]+)')
    def get_parameters(self, request, model_type=None):
        params = MODEL_PARAMETER_SCHEMA.get(model_type)
        if not params:
            return Response({"error": "Invalid model type"}, status=400)

        return Response({"parameters": list(params.keys())})
  
        
