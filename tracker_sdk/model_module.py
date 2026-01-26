import os
import pickle

class ModelModule:
    STORAGE_DIR = "storage/models"

    def __init__(self):
        os.makedirs(self.STORAGE_DIR, exist_ok=True)

    def save_model(self, model, experiment_id, name="model.pkl"):
        path = os.path.join(self.STORAGE_DIR, f"{experiment_id}_{name}")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return path
