from typing import Dict
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

DUMP_DIR = 'dump'


def get_rigid_body_transformation(state):
    state = np.array(state)
    P = np.zeros((4, 4))
    P[0, 0] = np.cos(state[2])
    P[1, 1] = np.cos(state[2])
    P[0, 1] = -np.sin(state[2])
    P[1, 0] = np.sin(state[2])
    P[2, 2] = 1
    P[:2, 3] = state[:2]
    P[3, 3] = 1
    return P

    
def get_relative_pose(true_pose, model_pose):
    return np.linalg.inv(true_pose) @ model_pose


def get_relative_poses_diff(true_pose1, true_pose2, model_pose1, model_pose2):
    return np.linalg.inv(np.linalg.inv(true_pose1) @ true_pose2) @ (np.linalg.inv(model_pose1) @ model_pose2)


def pose_error_trans(relative_poses : np.ndarray):
    N = relative_poses.shape[0]
    return (1./N * (np.linalg.norm(relative_poses[:, :3, 3], ord=2, axis=1)**2).sum(0))**.5


def pose_error_rot(relative_poses : np.ndarray):
    N = relative_poses.shape[0]
    trace = np.trace(relative_poses[:, :3, :3], axis1=1, axis2=2)
    return (1./N * (np.arccos((trace-1)/2)**2).sum(0))**.5


#def APE_trans(true_trans, model_trans):
#    pass
    #return np.sqrt(np.linalg.norm(model_states[:, :2] / true_states[:, :2]) ** 2 / len(true_states))



#def APE_rot(true_states, model_states):
#    pass
    #return np.sqrt(np.linalg.norm(model_states[:, 2] / true_states[:, 2]) ** 2 / len(true_states))


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

    
