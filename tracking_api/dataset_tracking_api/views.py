import hashlib
from django.shortcuts import render
from rest_framework.exceptions import ValidationError
from rest_framework import viewsets
from .models import Datasets
from .serializers import DatasetsSerializer
# Create your views here.
class DatasetsViewSet(viewsets.ModelViewSet):
    try:
        queryset = Datasets.objects.all().order_by('-created_at')
        serializer_class = DatasetsSerializer
    except Exception as e:
        print(f"No serialized class is recognized:{e}")

    try:
        def perform_create(self, serializer): #perform_create() is a built-in hook in DRF’s CreateModelMixin. It is called automatically on POST.
            fileObj = self.request.FILES['document']
            if not fileObj:
                raise ValidationError("No file uploaded.")

            try:
                hasher = hashlib.sha256()
                for chunk in fileObj.chunks():
                    hasher.update(chunk)
                file_hash = hasher.hexdigest()
            except:
                raise 

            latest = Datasets.objects.order_by('-version').first()
            version=1
            if latest:
                if latest.hash == file_hash:
                    raise ValueError("This dataset version is identical to existing version")
            
            version = latest.version +1

            serializer.save(hash=file_hash, version=version)
    except Exception as e:
        print(f"dataset hashing and version does not occur, can lead to duplicate data: {e}")


