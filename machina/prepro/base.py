import numpy as np
import torch


class BasePrePro(object):
    """
    Preprocess for observations.

    Parameters
    ----------
    observation_space : gym.Space
    normalize_ob : bool
    """

    def __init__(self, observation_space, normalize_ob=True):
        self.observation_space = observation_space
        self.normalize_ob = normalize_ob
        if self.normalize_ob:
            self.ob_rm = np.zeros(self.observation_space.shape)
            self.ob_rv = np.ones(self.observation_space.shape)
            self.alpha = 0.001

    def update_ob_rms(self, ob):
        """
        Updating running mean and running variance.
        """
        self.ob_rm = self.ob_rm * (1-self.alpha) + self.alpha * ob
        self.ob_rv = self.ob_rv * (1-self.alpha) + \
            self.alpha * np.square(ob-self.ob_rm)

    def prepro(self, ob):
        """
        Applying preprocess to observations.
        """
        if self.normalize_ob:
            ob = (ob - self.ob_rm) / (np.sqrt(self.ob_rv) + 1e-8)
            ob = np.clip(ob, -5, 5)
        return ob

    def prepro_with_update(self, ob):
        """
        Applying preprocess to observations with update.
        """
        if self.normalize_ob:
            self.update_ob_rms(ob)
            ob = (ob - self.ob_rm) / (np.sqrt(self.ob_rv) + 1e-8)
            ob = np.clip(ob, -5, 5)
        return ob
