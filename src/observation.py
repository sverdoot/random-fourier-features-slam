import numpy as np
from abc import ABC, abstractmethod


class ObservationModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_measurement(self, state, landmark):
        pass

    @abstractmethod
    def get_jacobian(self, state, landmark, measurement):
        pass


class RangeModel(ObservationModel):
    def __init__(self, range_noise_std, **kwargs):
        self.range_noise_std = range_noise_std

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


class RangeBearingModel(ObservationModel):
    def __init__(self, range_noise_std, bearing_noise_std, **kwargs):
        self.range_noise_std = range_noise_std
        self.bearing_noise_std = bearing_noise_std

    def get_measurement(self, state, landmark):
        h = np.sqrt((state[0] - landmark[0])**2 + (state[1] - landmark[1])**2)
        return h

    def get_jacobian(self, state, landmark, measurement):
        dh = np.array([(state[0] - landmark[0]) / measurement, 
                        (state[1] - landmark[1]) / measurement,
                        -(state[0] - landmark[0]) / measurement, 
                        -(state[1] - landmark[1]) / measurement])
        return dh

    def get_noisy_meausurement(self, state, landmark):
        return self.get_measurement(state, landmark) #+ self.range_noise_std * np.random.randn()