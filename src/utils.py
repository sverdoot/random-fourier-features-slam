from typing import Dict
import numpy as np
from pathlib import Path
import json
from collections import defaultdict


def APE_trans(true_states, model_states):
    return np.sqrt(np.linalg.norm(model_states[:, :2] / true_states[:, :2]) ** 2 / len(true_states))


def APE_rot(true_states, model_states):
    return np.sqrt(np.linalg.norm(model_states[:, 2] / true_states[:, 2]) ** 2 / len(true_states))


def save_result(model, metrics : dict, path_to_save, args):
    path = Path(path_to_save)
    if path.exists():
        with path.open('r') as f:
            storage = json.load(f)
    else:
        storage = defaultdict(dict)

    if str(args.num_landmarks) not in storage[args.obs_model].keys():
        storage[args.obs_model][str(args.num_landmarks)] = defaultdict(list)
    storage[args.obs_model][str(args.num_landmarks)][args.noise_level].append(metrics)

    with path.open('w') as f:
        json.dump(storage, f)

    
