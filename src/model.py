import torch
import numpy as np
import scipy
from scipy.linalg import block_diag
from pathlib import Path
from scipy.stats import multivariate_normal

from data_utils import get_prediction, wrap_angle
from random_features import RFF


class Model(object):
    def __init__(self,
                rff : RFF, 
                initial_values, 
                initial_state : np.ndarray,
                times, 
                measurements,
                measurement_covs, 
                b_means,
                b_covs,
                land_means,
                land_cov,
                observation_model : callable,
                prior_mean : callable = get_prediction,
                tol : float = 1e-3,
                dumpening_factor : float = 1e-2,
                max_n_iter : int = 1e3,
                verbose : bool = True,
                **kwargs):

        self.rff = rff
        self.initial_values = initial_values
        self.initial_state = initial_state
        self.times = np.array(times)
        self.measurements = measurements
        self.M = n_landmarks = len(land_cov)
        self.b = initial_values
        self.b_means = np.concatenate([b_means.reshape(-1), land_means.reshape(-1)], 0)
        #self.land_means = land_means
        self.max_n_iter = max_n_iter
        self.dampening_factor = dumpening_factor
        self.tol = tol
        self.verbose = verbose

        self.K_matrix = block_diag(*b_covs)
        self.P_matrix = block_diag(self.K_matrix, *land_cov)
        self.inv_P_matrix = block_diag(*[np.linalg.inv(x) for x in b_covs], *[np.linalg.inv(x) for x in land_cov])

        self.measurement_covs = measurement_covs
        self.inv_measurement_covs = [np.linalg.inv(x) for x in measurement_covs]
        self.R_matrix = block_diag(*measurement_covs)
        self.inv_R_matrix = block_diag(*[np.linalg.inv(x) for x in measurement_covs])

        self.n_features = 2*rff.n_features
        self.state_dim = len(b_covs)

        self.prior_mean = prior_mean
        self.observation_model = observation_model

        self.features = np.stack([self.rff.get_random_features(self.times) for _ in range(self.state_dim)], 1)

        self.prior_means = []
        self.states = np.zeros((len(times), 3))
        self.estimates_history = []

        self.landmark_dim = 2

        self.state_feature_dim = self.n_features * self.state_dim

    def update_params(self, iter_id=0):
        features = self.features
        A = np.zeros((self.state_feature_dim + self.M * self.landmark_dim, self.state_feature_dim + self.M * self.landmark_dim))
        g = np.zeros(self.state_feature_dim + self.M * self.landmark_dim)
        self.prior_means = []
        for i, t in enumerate(self.times):
            diag = np.concatenate([features[i].reshape(-1), np.ones(self.landmark_dim * self.M)], 0)
            phi_matrix = np.diag(diag)

            y_vector = phi_matrix @ self.b
            state_cut = y_vector[:self.state_feature_dim].reshape(self.state_dim, self.n_features).sum(1)
            if iter_id == 0:
                if i == 0:
                    pass
                else:
                    self.states[i-1] = self.prior_means[i-1]
            mean = self.prior_mean(t, np.concatenate([self.initial_state.reshape(1, self.state_dim), self.states], 0))
            self.prior_means.append(mean)
            state_cut += mean

            measurement = self.measurements[i][0]  # assume only one landmark per step
            landmark_id = int(measurement[-1])  # assume data assotiation is known
            observation = measurement[:-1]
            landmark_start_idx = self.state_feature_dim+landmark_id*self.landmark_dim
            landmark_cut = y_vector[landmark_start_idx:landmark_start_idx+self.landmark_dim]
            h = self.observation_model.get_measurement(state_cut, landmark_cut)
            dh = self.observation_model.get_jacobian(state_cut, landmark_cut, h)
            obs_dim = dh.shape[0]

            state_part = dh[:, :self.state_dim].reshape(obs_dim, self.state_dim, 1).repeat(self.n_features, 2).reshape(obs_dim, -1)
            
            lands_part = np.zeros((obs_dim, self.landmark_dim*self.M))
            lands_part[:, self.landmark_dim*landmark_id : self.landmark_dim*landmark_id + self.landmark_dim] = dh[:, self.state_dim:]
            
            dh = np.concatenate([state_part, lands_part], 1)

            A += phi_matrix.T @ dh.T @ self.inv_measurement_covs[i] @ dh @ phi_matrix
            
            diff = observation - h
            if hasattr(self.observation_model, 'bearing_noise_std'):
                diff[-1] = wrap_angle(diff[-1])
            
            g += phi_matrix.T @ dh.T @ self.inv_measurement_covs[i] @ diff

        A += self.inv_P_matrix
        diff = self.b - self.b_means
        #if self.state_dim == 3:
        #    diff[(self.state_dim-1)*self.n_features:self.state_dim*self.n_features+1] = wrap_angle(diff[(self.state_dim-1)*self.n_features:self.state_dim*self.n_features+1])
        g -= self.inv_P_matrix @ diff #(self.b - self.b_means)

        return A, g

    def solve(self, A, g):
        A = A + self.dampening_factor * np.diag(A)[:, None]
        #print('Conditional number:', np.linalg.cond(A))
        db = scipy.linalg.solve(A, g)
        return db

    def iteration(self, iter_id=0):
        A, g = self.update_params(iter_id=iter_id)
        db = self.solve(A, g)
        self.b += db
        #if self.state_dim == 3:
        #    self.b[(self.state_dim-1)*self.n_features:self.state_dim*self.n_features+1] = wrap_angle(self.b[(self.state_dim-1)*self.n_features:self.state_dim*self.n_features+1])

        self.states = self.prior_means + \
            (self.features * self.b[:self.state_feature_dim].reshape(self.state_dim, -1)).sum(-1)
        if self.state_dim == 3:
            self.states[:, 2] = wrap_angle(self.states[:, 2])
        self.estimates_history.append(self.states)
        return np.linalg.norm(db, 2)

    def run_slam(self, n_iter=None):
        eps = float('inf')
        n = 0
        while eps > self.tol and n < self.max_n_iter:
            eps = self.iteration(iter_id=n)
            if self.verbose and n % 10 == 0:
                print(f'Iter {n}. Norm of perturbation: {eps:.4f}')
            n += 1
            if n_iter is not None and n >= n_iter:
                if self.verbose:
                    print(f'Finished {n_iter} iterations')
                break
        if  n >= self.max_n_iter:
            if self.verbose:
                print(f'Exceeded max number of iterations: {self.max_n_iter}')


def train(
        landscape,
        odometry,
        observation_model,
        n_features = 100,
        sigma_l = 3.,
        n_iter = 100,
        b_sigma = 0.1,
        land_sigma = 3,
        dampening_factor = 0.05,
        landmark_mean='true',
        verbose=False
        ):

    state_dim = odometry.state_dim
    landmark_dim = landscape.landmark_dim
    obs_dim = observation_model.obs_dim

    num_landmarks = landscape.num_landmarks
    num_points = odometry.num_points

    sampler = lambda shape: multivariate_normal(mean = [0], 
                                                cov = (1./sigma_l**2) * np.eye(1)
                                                ).rvs(size = np.product(shape))
    assert n_features % 2 == 0
    rff = RFF(n_features = n_features // 2, w_sampler=sampler, dim=1)

    def prior_mean(t, state_estimates):
        i = odometry.times.index(t)
        state = get_prediction(state_estimates[i], odometry.motions[i])
        return state

    initial_values = np.zeros((n_features*state_dim)+num_landmarks*landmark_dim)
    b_means = np.zeros(n_features*state_dim)
    if landmark_mean == 'true':
        land_means = landscape.landmarks
    elif landmark_mean == 'zero':
        land_means = np.zeros((num_landmarks, landmark_dim))
    initial_values[n_features*state_dim:] = land_means.reshape(-1)

    observations = observation_model.filter_odom_observations(odometry.observations)
    Q = observation_model.Q

    model = Model(rff=rff, 
                initial_values=initial_values,
                initial_state=odometry.states[0],
                times=odometry.times,
                measurements=observations,
                measurement_covs=[Q] * num_points,
                motions=odometry.motions,
                b_means=b_means,
                b_covs=[b_sigma**2 * np.eye(n_features)] * state_dim,
                land_means=land_means,
                land_cov=[land_sigma**2 * np.eye(landmark_dim)] * num_landmarks,
                observation_model=observation_model,
                prior_mean=prior_mean,
                dampening_factor=dampening_factor,
                verbose=verbose, 
                tol=1e-5
                )

    model.run_slam(n_iter=n_iter)
    return model