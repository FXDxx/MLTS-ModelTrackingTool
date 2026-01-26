from rest_framework.routers import DefaultRouter
from .views import MLModelViewSet

router = DefaultRouter()
router.register(r'models', MLModelViewSet, basename='model')
urlpatterns = router.urls