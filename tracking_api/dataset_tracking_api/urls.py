from rest_framework.routers import DefaultRouter
from .views import DatasetsViewSet


router = DefaultRouter()
router.register(r'datasets', DatasetsViewSet, basename='dataset')
urlpatterns = router.urls