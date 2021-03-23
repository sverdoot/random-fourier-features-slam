import torch
import numpy as np
import scipy
from scipy.linalg import block_diag
from pathlib import Path

from data_utils import get_prediction, wrap_angle


class Model(object):
    def __init__(self,
                rff, 
                initial_values, 
                initial_state,
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
                dumpening_factor : float =1e-2,
                max_n_iter=1e3,
                **kwargs):

        self.rff = rff
        self.initial_values = initial_values
        self.initial_state = initial_state
        self.times = times
        self.measurements = measurements
        self.tol = tol
        self.max_n_iter = max_n_iter
        self.M = n_landmarks = len(land_means)
        self.b = initial_values
        self.land_means = land_means
        self.b_means = b_means 
        #self.motions = motions

        self.K_matrix = block_diag(*b_covs)
        self.P_matrix = block_diag(self.K_matrix, *land_cov)
        self.inv_P_matrix = block_diag(*[np.linalg.inv(x) for x in b_covs], *[np.linalg.inv(x) for x in land_cov]) #np.linalg.inv(land_cov))

        self.measurement_covs = measurement_covs
        self.inv_measurement_covs = [np.linalg.inv(x) for x in measurement_covs]
        self.R_matrix = block_diag(*measurement_covs)
        self.inv_R_matrix = block_diag(*[np.linalg.inv(x) for x in measurement_covs])

        self.n_features = 2*rff.n_features
        self.state_dim = len(b_covs)
        #print(self.state_dim)

        self.prior_mean = prior_mean

        self.observation_model = observation_model

        self.dampening_factor = dumpening_factor

        self.times = np.array(self.times)
        self.features = np.stack([self.rff.get_random_features(self.times) for _ in range(self.state_dim)], 1)

        self.prior_means = []
        self.states = np.zeros((len(times), 3))


        self.estimates_history = []

    def update_params(self, n_iter=0):
        #features = np.stack([self.rff.get_random_features(self.times) for _ in range(self.state_dim)], 1)
        features = self.features

        A = 0
        g = 0
        self.prior_means = []
        for i, t in enumerate(self.times):
            diag = np.concatenate([features[i].reshape(-1), np.ones(2*self.M)], 0)
            phi_matrix = np.diag(diag)


            y_vector = phi_matrix @ self.b
            #y_vector[:self.n_features*self.state_dim] += self.prior_mean
            cut = y_vector[:self.n_features*self.state_dim].reshape(self.state_dim, self.n_features).sum(1)
            if n_iter <  10: # == 0 (!)
                if i == 0:
                    pass
                else:
                    self.states[i-1] = self.prior_means[i-1]
            mean = self.prior_mean(t, np.concatenate([self.initial_state.reshape(1, self.state_dim), self.states], 0))
            self.prior_means.append(mean)
            cut += mean
            #reconst = np.concatenate([cut, y_vector[self.n_features*self.state_dim:]], 0)

            landmark_id = int(self.measurements[i, -1])
            h = self.observation_model.get_measurement(cut, y_vector[self.n_features*self.state_dim+landmark_id*2:self.n_features*self.state_dim+landmark_id*2+2])
            dh = self.observation_model.get_jacobian(cut, y_vector[self.n_features*self.state_dim+landmark_id*2:self.n_features*self.state_dim+landmark_id*2+2], h)
            obs_dim = dh.shape[0]

            state_part = dh[:, :self.state_dim].reshape(obs_dim, self.state_dim, 1).repeat(self.n_features, 2).reshape(obs_dim, -1)
            
            lands_part = np.zeros((obs_dim, 2*self.M))

            lands_part[:, 2*landmark_id:2*landmark_id+2] = dh[:, self.state_dim:]
            
            dh = np.concatenate([state_part, lands_part], 1)

            A += phi_matrix.T @ dh.T @ self.inv_measurement_covs[i] @ dh @ phi_matrix

            g += phi_matrix.T @ dh.T @ self.inv_measurement_covs[i] @ (self.measurements[i, :-1] - h)

        A += self.inv_P_matrix
        g += self.inv_P_matrix @ (self.b - self.b_means)

        return A, g

    def solve(self, A, g):
        A = A + self.dampening_factor * np.diag(A)[:, None]
        db = scipy.linalg.solve(A, g)
        return db

    def iteration(self, n_iter=0):
        A, g = self.update_params(n_iter=n_iter)
        db = self.solve(A, g)
        self.b += db
        self.states = self.prior_means + (self.features * self.b[:self.n_features*self.state_dim].reshape(self.state_dim, -1)).sum(-1)
        self.states[:, 2] = wrap_angle(self.states[:, 2])
        self.estimates_history.append(self.states)
        return np.linalg.norm(db, 2)

    def run_slam(self, n_iter=None):
        eps = 10
        n = 0
        while eps > self.tol and n < self.max_n_iter:
            eps = self.iteration(n_iter=n)
            print(eps)
            n += 1
            if n_iter is not None and n >= n_iter:
                break