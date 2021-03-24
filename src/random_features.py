import numpy as np


class RFF(object):
    def __init__(self, n_features, w_sampler: callable, dim=1, **kwargs):
        self.w_sampler = w_sampler
        self.dim = dim
        self.n_features = n_features

    def sample_vectors(self, n_vectors=1):
        shape = [n_vectors, self.n_features]
        vectors = self.w_sampler(shape).reshape(n_vectors, self.n_features, self.dim)
        return vectors

    def get_random_features(self, pts):
        pts = pts.reshape(-1, self.dim)
        vectors = self.sample_vectors(n_vectors=len(pts))
        dots = (vectors * pts[:, None, :]).sum(-1)
        features = np.concatenate([np.cos(dots), np.sin(dots)], 1)
        return features
