from .dataset_module import DatasetModule
from .experiment_module import ExperimentModule
from .model_module import ModelModule
from .metric_module import MetricModule
from .api_client import APIClient

class Tracker:
    def __init__(self, api_url="http://localhost:8000"):
        self.dataset_module = DatasetModule()
        self.experiment_module = ExperimentModule()
        self.model_module = ModelModule()
        self.metric_module = MetricModule()
        self.api_client = APIClient(api_url)

        self.current_dataset_id = None
        self.current_experiment_id = None

    def log_dataset(self, X, y, name="dataset"):
        self.current_dataset_id = self.dataset_module.register_dataset(X, y, name)
        return self.current_dataset_id

    def start_experiment(self, model_name):
        if not self.current_dataset_id:
            raise Exception("Dataset must be logged first")
        self.current_experiment_id = self.experiment_module.start_experiment(
            model_name, self.current_dataset_id
        )
        return self.current_experiment_id

    def log_hyperparameters(self, params: dict):
        if not self.current_experiment_id:
            raise Exception("Experiment must be started first")
        self.experiment_module.log_hyperparameters(self.current_experiment_id, params)

    def log_metric(self, name, value):
        if not self.current_experiment_id:
            raise Exception("Experiment must be started first")
        self.metric_module.log_metric(self.current_experiment_id, name, value)

    def save_model(self, model, name="model.pkl"):
        if not self.current_experiment_id:
            raise Exception("Experiment must be started first")
        self.model_module.save_model(model, self.current_experiment_id, name)
