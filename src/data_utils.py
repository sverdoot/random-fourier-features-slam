import numpy as np


def sense_landmarks(state, field_map, max_observations):
    """
    Observes num_observations of landmarks for the current time step.
    The observations will be in the front plan of the robot.

    :param state: The current state of the robot (format: np.array([x, y, theta])).
    :param field_map: The FieldMap object. This is necessary to extract the true landmark positions in the field.
    :param max_observations: The maximum number of observations to generate per time step.
    :return: np.ndarray or size num_observations x 3. Each row is np.array([range, bearing, lm_id]).
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(field_map, Landscape)

    assert state.shape == (3,)

    M = field_map.num_landmarks
    noise_free_observations_list = list()
    for k in range(M):
        noise_free_observations_list.append(get_observation(state, field_map, k))
    noise_free_observation_tuples = [(x[0], np.abs(x[1]), int(x[2])) for x in noise_free_observations_list]

    dtype = [('range', float), ('bearing', float), ('lm_id', int)]
    noise_free_observations = np.array(noise_free_observations_list)
    noise_free_observation_tuples = np.array(noise_free_observation_tuples, dtype=dtype)

    ii = np.argsort(noise_free_observation_tuples, order='bearing')
    noise_free_observations = noise_free_observations[ii]
    noise_free_observations[:, 2] = noise_free_observations[:, 2].astype(int)

    c1 = noise_free_observations[:, 1] > -np.pi / 2.
    c2 = noise_free_observations[:, 1] <  np.pi / 2.
    ii = np.nonzero((c1 & c2))[0]

    if ii.size <= max_observations:
        return noise_free_observations[ii]
    else:
        return noise_free_observations[:max_observations]

class Landscape(object):
    def __init__(self, num_landmarks=10, field_borders=(-10, 10)):
        self.num_landmarks = num_landmarks
        self.field_borders = field_borders

        self.landmarks = []
        self.landmarks_poses_x = []
        self.landmarks_poses_y = []

    def initialize(self):
        self.landmarks = np.random.uniform(self.field_borders[0], self.field_borders[1], (self.num_landmarks, 2)).reshape(self.num_landmarks, 2)
        self.landmarks_poses_x = self.landmarks[:, 0]
        self.landmarks_poses_y = self.landmarks[:, 1]
        

class Odometry(object):
    def __init__(self, landscape, observation_dim, max_time=100):
        self.landscape = landscape
        self.states = []
        self.observations = []
        self.motions = []
        self.times = []

        self.alphas = np.array([0.05, 0.001, 0.05, 0.01])
        self.beta = np.array([10., 10.])
        self.Q = np.diag([*(self.beta ** 2), 0])

        self.observation_dim = observation_dim
        self.max_time = max_time

    def generate(self, n_steps):
        self.times = sorted(np.random.uniform(0, self.max_time, n_steps))
        initial_state = np.random.randn(3)
        state = initial_state
        drot1 = 0
        dtr = 1
        drot2 = 0
        for i in range(n_steps):
            noisy_observations = []
            while len(noisy_observations) == 0:
                new_drot1 = np.random.normal(drot1, 0.05)
                new_dtr = np.random.normal(dtr, 0.05)
                new_drot2 = np.random.normal(drot2, 0.05)

                motion = np.array([new_drot1, new_dtr, new_drot2])
                
                new_state = sample_from_odometry(state, motion, self.alphas)
                
                noise_free_observations = sense_landmarks(new_state, self.landscape, 1)
                observation_noise = np.random.multivariate_normal(np.zeros(self.observation_dim), self.Q)
                # Generate noisy observation as observed by the robot for the filter.
                noisy_observations = np.empty(observation_noise.shape)
                #print(noisy_observations.shape, observation_noise.shape)
                #noisy_observations[0] = noise_free_observations[0] + observation_noise
                noisy_observations = noise_free_observations + observation_noise
            state = new_state
            drot1 = new_drot1
            dtr = new_dtr
            drot2 = new_drot2
            self.states.append(state)
            self.motions.append(motion)
            self.observations.append(noisy_observations)



def sample_from_odometry(state, motion, alphas):
    """
    Predicts the next state (a noisy version) given the current state, and the motion command.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param motion: The motion command (format: [drot1, dtran, drot2]) to execute.
    :param alphas: The motion noise parameters (format: [a1, a2, a3, a4]).
    :return: A noisy version of the state prediction (format: [x, y, theta]).
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(motion, np.ndarray)

    assert state.shape == (3,)
    assert motion.shape == (3,)
    assert len(alphas) == 4

    a1, a2, a3, a4 = alphas
    drot1, dtran, drot2 = motion
    noisy_motion = np.zeros(motion.size)

    noisy_motion[0] = np.random.normal(drot1, np.sqrt(a1 * (drot1 ** 2) + a2 * (dtran ** 2)))
    noisy_motion[1] = np.random.normal(dtran, np.sqrt(a3 * (dtran ** 2) + a4 * ((drot1 ** 2) + (drot2 ** 2))))
    noisy_motion[2] = np.random.normal(drot2, np.sqrt(a1 * (drot2 ** 2) + a2 * (dtran ** 2)))

    return get_prediction(state, noisy_motion)


def get_observation(state, field_map, lm_id):
    """
    Generates a sample observation given the current state of the robot and the marker id of which to observe.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param field_map: A map of the field.
    :param lm_id: The landmark id indexing into the landmarks list in the field map.
    :return: The observation to the landmark (format: np.array([range, bearing, landmark_id])).
             The bearing (in rad) will be in [-pi, +pi].
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(field_map, Landscape)

    assert state.shape == (3,)

    lm_id = int(lm_id)

    dx = field_map.landmarks_poses_x[lm_id] - state[0]
    dy = field_map.landmarks_poses_y[lm_id] - state[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)
    bearing = np.arctan2(dy, dx) - state[2]

    return np.array([distance, wrap_angle(bearing), lm_id])


def wrap_angle(angle):
    """
    Wraps the given angle to the range [-pi, +pi].

    :param angle: The angle (in rad) to wrap (can be unbounded).
    :return: The wrapped angle (guaranteed to in [-pi, +pi]).
    """

    pi2 = 2 * np.pi

    while angle < -np.pi:
        angle += pi2

    while angle >= np.pi:
        angle -= pi2

    return angle

    
def get_prediction(state, motion):
    """
    Predicts the next state given state and the motion command.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param motion: The motion command to execute (format: [drot1, dtran, drot2]).
    :return: The next state of the robot after executing the motion command
             (format: np.array([x, y, theta])). The angle will be in range
             [-pi, +pi].
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(motion, np.ndarray)

    assert state.shape == (3,)
    assert motion.shape == (3,)

    x, y, theta = state
    drot1, dtran, drot2 = motion

    theta += drot1
    x += dtran * np.cos(theta)
    y += dtran * np.sin(theta)
    theta += drot2

    # Wrap the angle between [-pi, +pi].
    theta = wrap_angle(theta)

    return np.array([x, y, theta])


