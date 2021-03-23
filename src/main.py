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
from utils import APE_rot, APE_trans, save_result



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_model', help='observation model', type=str, choices=["range-bearing", "range", "bearing"], default="range-bearing")
    parser.add_argument('--beta', '-b', dest='beta',
                        help='std of observations. if model is range-bearing range first, bearing second',
                        type=float, nargs='+', default=np.array([2., 5.]))
    parser.add_argument('-M', '--num_landmarks', dest='num_landmarks', type=int, default=20)
    parser.add_argument('-N', '--num_points', dest='num_points', type=int, default=100)
    parser.add_argument('-D', '--num_features', dest='num_features', type=int, default=100)
    parser.add_argument('--landmark_mean', type=str, choices=['true', 'zero'], default='true')
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--path_to_trajectory', type=str, default=None)
    parser.add_argument('--path_to_dump', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)

    beta = copy.copy(args.beta)
    if args.obs_model == 'bearing':
        beta[0] = np.deg2rad(args.beta[0])
    elif args.obs_model == 'range-bearing':
        beta[1] = np.deg2rad(args.beta[1])

    observation_model = ObservationModelFactory(args.obs_model, *beta)

    landscape = Landscape()

    if args.path_to_trajectory is not None:
        landscape.load(Path(args.path_to_trajectory))
        odometry = Odometry(landscape=landscape)
        odometry.load(Path(args.path_to_trajectory))
    else:
        landscape.initialize()
        odometry = Odometry(landscape=landscape, beta=beta)
        odometry.generate(n_steps=args.num_points)

    model = train(landscape, 
            odometry, 
            observation_model,
            n_features=args.num_features,
            n_iter=args.n_iter,
            landmark_mean=args.landmark_mean,
            verbose=args.verbose
            )
    estimated_states = np.stack(model.states, 0)
    real_sates = np.stack(odometry.states[1:], 0)

    ape_rot = APE_rot(real_sates, estimated_states)
    ape_trans = APE_trans(real_sates, estimated_states)

    if args.verbose:
        print(f'APE rot: {ape_rot:.4f}, APE trans: {ape_trans:.4f}')


    if args.path_to_dump is not None:
        args.noise_level = str(tuple(args.beta))
        metrics = {"ape_trans" : ape_trans, "ape_rot" : ape_rot}
        save_result(model, metrics, args.path_to_dump, args)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

