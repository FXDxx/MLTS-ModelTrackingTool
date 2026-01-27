from rest_framework.routers import DefaultRouter
from .views import DatasetsViewSet

try:
    router = DefaultRouter()
except Exception as e:
    print(f"Error in creating DefaultRouter object: {e}")

try:
    router.register(r'datasets', DatasetsViewSet, basename='dataset')
except Exception as e:
    print(f"Invalid URL:{e}")

try:
    urlpatterns = router.urls
except Exception as e:
    print(f"Router not recognized")