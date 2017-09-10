import numpy as np
import torch

class BasePrePro(object):
    def __init__(self, ob_space, normalize_obs=True):
        self.ob_space = ob_space
        self.normalize_obs = normalize_obs
        if self.normalize_obs:
            self.ob_rm = np.zeros(self.ob_space.shape)
            self.ob_rv = np.ones(self.ob_space.shape)
            self.alpha = 0.001

    def update_ob_rms(self, ob):
        self.ob_rm = self.ob_rm * (1-self.alpha) + self.alpha * ob
        self.ob_rv = self.ob_rv * (1-self.alpha) + self.alpha * np.square(ob-self.ob_rm)

    def prepro(self, obs):
        normalized_obs = (obs - self.ob_rm) / (np.sqrt(self.ob_rv) + 1e-8)
        return np.clip(normalized_obs, -5, 5)

    def prepro_with_update(self, ob):
        normalized_ob = (ob - self.ob_rm) / (np.sqrt(self.ob_rv) + 1e-8)
        self.update_ob_rms(ob)
        return np.clip(normalized_ob, -5, 5)


