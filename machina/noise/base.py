

class BaseActionNoise(object):
    """
    Base class of action noise.
    """

    def __init__(self, ac_space):
        self.ac_space = ac_space

    def reset(self):
        pass
