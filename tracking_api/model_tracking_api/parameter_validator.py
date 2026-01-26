from .ml_registry import MODEL_PARAMETER_SCHEMA

def validate_parameters(model_type, user_params):
    schema = MODEL_PARAMETER_SCHEMA.get(model_type)

    if not schema:
        raise ValueError("Unsupported model type")

    validated = {}

    for key, value in user_params.items():
        if key not in schema:
            raise ValueError(f"{key} not allowed for {model_type}")

        expected_type = schema[key]

        try:
            validated[key] = expected_type(value)
        except:
            raise ValueError(f"{key} must be {expected_type.__name__}")

    return validated
