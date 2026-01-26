class ExperimentModule:
    def __init__(self):
        self.experiments = {}  # experiment_id: {model_name, dataset_id, hyperparams}
        self.counter = 0

    def start_experiment(self, model_name, dataset_info):
        self.counter += 1
        exp_id = self.counter
        self.experiments[exp_id] = {
            "model_name": model_name,
            "dataset_id": dataset_info,
            "hyperparameters": {},
        }
        return exp_id

    def log_hyperparameters(self, experiment_id, params: dict):
        self.experiments[experiment_id]["hyperparameters"] = params
