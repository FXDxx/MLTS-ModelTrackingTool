import requests

class APIClient:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url

    def post_dataset(self, dataset_info):
        return requests.post(f"{self.api_url}/model-save/datasets/", json=dataset_info)

    def post_experiment(self, experiment_info):
        return requests.post(f"{self.api_url}/model-save/experiments/", json=experiment_info)

    def post_hyperparameters(self, hyperparams_info):
        return requests.post(f"{self.api_url}/model-save/hyperparameters/", json=hyperparams_info)

    def post_metric(self, metric_info):
        return requests.post(f"{self.api_url}/model-save/metrics/", json=metric_info)