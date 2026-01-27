from rest_framework.routers import DefaultRouter
from .views import ModelInferenceViewset

try:
    router = DefaultRouter()
except Exception as e:
    print(f"Error initializing DefaultRouter object: {e}")
try:
    router.register(r'inferences', ModelInferenceViewset, basename='inference')
except Exception as e:
    print(f"Error registering ModelInferenceViewset: {e}")
try:
    urlpatterns = router.urls
except Exception as e:
    print(f"Error getting router URLs: {e}")