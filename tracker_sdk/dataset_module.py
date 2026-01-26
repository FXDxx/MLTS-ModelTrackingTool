import hashlib
import os
import pickle

class DatasetModule:
    STORAGE_DIR = "storage/datasets"

    def __init__(self):
        os.makedirs(self.STORAGE_DIR, exist_ok=True)
        self.dataset_versions = {}  # {hash: version}

    def _hash_dataset(self, X, y):
        data_bytes = pickle.dumps((X, y))
        return hashlib.sha256(data_bytes).hexdigest()

    def register_dataset(self, X, y, name="dataset"):
        dataset_hash = self._hash_dataset(X, y)
        if dataset_hash not in self.dataset_versions:
            version = len(self.dataset_versions) + 1
            self.dataset_versions[dataset_hash] = version
            # Save locally
            with open(os.path.join(self.STORAGE_DIR, f"{dataset_hash}.pkl"), "wb") as f:
                pickle.dump((X, y), f)
        else:
            version = self.dataset_versions[dataset_hash]
        return {"name": name, "hash": dataset_hash, "version": version}
