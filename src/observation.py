import numpy as np
from abc import ABC, abstractmethod


class ObservationModel(ABC):
    @abstractmethod
    def get_measurement(self, state, landmark):
        pass

    @abstractmethod
    def get_jacobian(self, state, landmark, measurement):
        pass


class RangeModel(ObservationModel):
    obs_dim = 1
    def __init__(self, range_noise_std, *args, **kwargs):
        self.range_noise_std = range_noise_std
        self.Q = np.array([[range_noise_std]])

    def get_measurement(self, state, landmark):
        h = np.sqrt((state[0] - landmark[0])**2 + (state[1] - landmark[1])**2)
        return h

    def get_jacobian(self, state, landmark, measurement):
        dh = np.array([(state[0] - landmark[0]) / measurement, 
                        (state[1] - landmark[1]) / measurement,
                        0,
                        -(state[0] - landmark[0]) / measurement, 
                        -(state[1] - landmark[1]) / measurement])
        return dh.reshape(1, -1)

    def get_noisy_meausurement(self, state, landmark):
        return self.get_measurement(state, landmark) + self.range_noise_std * np.random.randn()

    def filter_odom_observations(self, observations):
        return [x[:, [0, 2]] for x in observations]


class BearingModel(ObservationModel):
    obs_dim = 1
    def __init__(self, bearing_noise_std, *args, **kwargs):
        self.bearing_noise_std = bearing_noise_std
        self.Q = np.array([[bearing_noise_std]])

    def get_measurement(self, state, landmark):
        h = np.array(
                [np.arctan2(landmark[0] - state[0], landmark[1] - state[1]) - state[2]]
        )
        return h

    def get_jacobian(self, state, landmark, measurement):
        dh = np.array(
                    [
                        [(landmark[1] - state[1]) / measurement[0]**2,
                        -(landmark[0] - state[0]) / measurement[0]**2,
                        -1.,
                        -(landmark[1] - state[1]) / measurement[0]**2,
                        (landmark[0] - state[0]) / measurement[0]**2
                        ]
                    ]
        )
        return dh

    def filter_odom_observations(self, observations):
        return [x[:, [1, 2]] for x in observations]


class RangeBearingModel(ObservationModel):
    obs_dim = 2
    def __init__(self, range_noise_std, bearing_noise_std, **kwargs):
        self.range_noise_std = range_noise_std
        self.bearing_noise_std = bearing_noise_std

        self.Q = np.diag((range_noise_std, bearing_noise_std))

    def get_measurement(self, state, landmark):
        h = np.array(
                [np.sqrt((state[0] - landmark[0])**2 + (state[1] - landmark[1])**2),
                np.arctan2(landmark[0] - state[0], landmark[1] - state[1]) - state[2]]
        )
        return h

    def get_jacobian(self, state, landmark, measurement):
        dh = np.array(
                    [
                        [(state[0] - landmark[0]) / measurement[0], 
                        (state[1] - landmark[1]) / measurement[0],
                        0,
                        -(state[0] - landmark[0]) / measurement[0], 
                        -(state[1] - landmark[1]) / measurement[0]],
                        
                        [(landmark[1] - state[1]) / measurement[0]**2,
                        -(landmark[0] - state[0]) / measurement[0]**2,
                        -1.,
                        -(landmark[1] - state[1]) / measurement[0]**2,
                        (landmark[0] - state[0]) / measurement[0]**2
                        ]
                    ]
        )
        return dh

    def get_noisy_meausurement(self, state, landmark):
        raise NotImplementedError

    def filter_odom_observations(self, observations):
        return observations


class ObservationModelFactory(object):
    models = [RangeModel, BearingModel, RangeBearingModel]
    def __new__(self, model, *args, **kwargs):
        self.model = model
        if model == 'range':
            return RangeModel(*args, **kwargs)
        elif model == 'bearing':
            return BearingModel(*args, **kwargs)
        elif model == 'range-bearing':
            return RangeBearingModel(*args, **kwargs)
        else:
            raise KeyError(f'no such observation model: {model}')