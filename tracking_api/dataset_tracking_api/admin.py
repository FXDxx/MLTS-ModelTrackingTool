from django.contrib import admin
from .models import Datasets
# Register your models here.
try:
    admin.site.register(Datasets)
except Exception as e:
    print(f"Datasets model not registered on admin")