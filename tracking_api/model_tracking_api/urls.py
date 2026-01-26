from rest_framework.routers import DefaultRouter
from .views import (DatasetsViewSet, ExperimentViewSet,
                    HyperparametersViewSet, MetricsViewSet, ModelArtifactsViewSet)

router = DefaultRouter()
router.register(r'models', ExperimentViewSet, basename='model')
router.register(r'hyperparameters', HyperparametersViewSet, basename='hyperparameter')
router.register(r'metrics', MetricsViewSet, basename='metric')
urlpatterns = router.urls