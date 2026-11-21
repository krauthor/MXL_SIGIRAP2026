import os
import json

import numpy as np
import torch


def store_results(results, filepath):
    print(f"Storing results at {filepath}...")

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, torch.Tensor):
                obj = obj.detach().cpu()
                return obj.item() if obj.numel() == 1 else obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(filepath, "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

    print(f"Results saved to {filepath}")
