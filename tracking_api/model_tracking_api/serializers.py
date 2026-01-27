from rest_framework import serializers
from .models import MLModels
from .parameter_validator import validate_parameters

class MLModelsSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModels
        fields = ['model_name', 'model_file','parameters','version','uploaded_at']
        read_only_fields = ['uploaded_at', 'parameters', 'version']

    def validate(self, data):
        model_type = data.get('model_name')
        param = data.get('parameters')
        if param:
            data["parameters"] = validate_parameters(model_type, param)

        return data
    
    