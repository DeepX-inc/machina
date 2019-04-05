

class BaseActionNoise(object):
    """
    Base class of action noise.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self):
        pass
