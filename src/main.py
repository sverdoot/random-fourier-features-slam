import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import argparse
from pathlib import Path
import copy

from random_features import RFF
from data_utils import Landscape, Odometry
from model import Model, train
from observation import ObservationModelFactory
from utils import (
    get_rigid_body_transformation, 
    get_relative_pose,
    get_relative_poses_diff, 
    pose_error_rot, 
    pose_error_trans, 
    save_result,
    DUMP_DIR
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_model', help='observation model', type=str, choices=["range-bearing", "range", "bearing"], default="range-bearing")
    parser.add_argument('--beta', '-b', dest='beta',
                        help='std of observations. if model is range-bearing range first, bearing second. Bearing noise should be in grad',
                        type=float, nargs='+', default=np.array([2., 5.]))
    parser.add_argument('-M', '--num_landmarks', dest='num_landmarks', type=int, default=20)
    parser.add_argument('-N', '--num_points', dest='num_points', type=int, default=100)
    parser.add_argument('-D', '--num_features', dest='num_features', type=int, default=100)
    parser.add_argument('--landmark_mean', type=str, choices=['true', 'zero'], default='zero')
    parser.add_argument('--lengthscale', type=float, default=3.)
    parser.add_argument('--b_sigma', type=float, default=0.01)
    parser.add_argument('--land_sigma', type=float, default=5)
    parser.add_argument('--dampening_factor', type=float, default=1e-5)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--path_to_trajs', type=str, default=None)
    parser.add_argument('--traj_id', type=int, default=0)
    parser.add_argument('--path_to_dump', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('--plot_traj', action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)

    beta = copy.copy(args.beta)
    if args.obs_model == 'bearing':
        beta[0] = np.deg2rad(args.beta[1])
        beta[1] = np.deg2rad(args.beta[1])
    elif args.obs_model == 'range-bearing':
        beta[1] = np.deg2rad(args.beta[1])

    observation_model = ObservationModelFactory(args.obs_model, *beta)

    landscape = Landscape(args.num_landmarks)

    if args.path_to_trajs is not None:
        args.traj_id = int(args.traj_id)
        trajs = np.load(Path(args.path_to_trajs))
        lands = trajs['lands'][args.traj_id]
        states = trajs['states'][args.traj_id]
        times = trajs['times'][args.traj_id]
        motions = trajs['motions'][args.traj_id]
        landscape.load_and_initialize(lands)
        odometry = Odometry(landscape=landscape, beta=np.array(beta))
        odometry.load_and_get_observations(states, times, motions)
    else:
        landscape.initialize()
        odometry = Odometry(landscape=landscape, beta=np.array(beta))
        odometry.generate(n_steps=args.num_points)

    model = train(landscape, 
            odometry, 
            observation_model,
            n_features=args.num_features,
            sigma_l=args.lengthscale,
            b_sigma=args.b_sigma,
            land_sigma=args.land_sigma,
            dampening_factor=args.dampening_factor,
            n_iter=args.n_iter,
            landmark_mean=args.landmark_mean,
            verbose=args.verbose
            )
    model_states = np.stack(model.states, 0)
    real_states = np.stack(odometry.states[1:], 0)

    # model_states[:, :2] /= 10
    # real_states[:, :2] /= 10
    # landscape.landmarks /= 10

    model_poses = [get_rigid_body_transformation(state) for state in model_states]
    real_poses = [get_rigid_body_transformation(state) for state in real_states]

    relative_poses = np.stack([get_relative_pose(r, m) for r, m in zip(real_poses, model_poses)], 0)
    relative_poses_diffs = np.stack([get_relative_poses_diff(
                                        real_poses[i], 
                                        real_poses[i+1], 
                                        model_poses[i], 
                                        model_poses[i+1]) for i in range(len(model_poses)-1)], 0)

    ape_rot = pose_error_rot(relative_poses)
    ape_trans = pose_error_trans(relative_poses)

    rpe_rot = pose_error_rot(relative_poses_diffs)
    rpe_trans = pose_error_trans(relative_poses_diffs)

    if args.verbose:
        print(f'APE rot: {ape_rot:.4f}, APE trans: {ape_trans:.4f}')
        print(f'RPE rot: {rpe_rot:.4f}, RPE trans: {rpe_trans:.4f}')

    if args.path_to_dump is not None:
        args.noise_level = str(tuple(args.beta))
        metrics = {"ape_trans" : ape_trans, "ape_rot" : ape_rot,
                    "rpe_trans" : rpe_trans, "rpe_rot" : rpe_rot}
        save_result(model, metrics, args.path_to_dump, args)

    if args.plot_traj:
        fig = plt.subplots()
        lands = np.stack(landscape.landmarks, 0)
        plt.scatter(lands[:, 0], lands[:, 1], c='b', s=150, alpha=0.7, label='true landmarks')
        #real_states = np.stack(odometry.states, 0)
        plt.plot(real_states[:, 0], real_states[:, 1], '-o', c='r', alpha=0.7, label='true trajectory')
        #real_states = np.stack(odometry.states, 0)
        model_lands = model.b[model.state_feature_dim:].reshape(args.num_landmarks, 2)
        plt.scatter(model_lands[:, 0], model_lands[:, 1], c='tab:gray', s=150, alpha=0.7, label='estimated landmarks')
        plt.plot(model_states[:, 0], model_states[:, 1], '-o', c='g', alpha=0.7, label='estimated trajectory')
        plt.legend()
        plt.savefig(Path(DUMP_DIR, 'traj.pdf'))
        plt.close()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

