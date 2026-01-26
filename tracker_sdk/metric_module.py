class MetricModule:
    def __init__(self):
        self.metrics = {}  # experiment_id: [{name, value}]

    def log_metric(self, experiment_id, name, value):
        if experiment_id not in self.metrics:
            self.metrics[experiment_id] = []
        self.metrics[experiment_id].append({"name": name, "value": value})
