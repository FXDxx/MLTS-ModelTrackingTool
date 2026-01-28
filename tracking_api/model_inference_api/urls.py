from rest_framework.routers import DefaultRouter
from .views import ModelInferenceViewset

router = DefaultRouter()
router.register(r'inferences', ModelInferenceViewset, basename='inference')
urlpatterns = router.urls
