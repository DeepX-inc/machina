# Copyright 2018 DeepX Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch


class BasePrePro(object):
    def __init__(self, ob_space, normalize_ob=True):
        self.ob_space = ob_space
        self.normalize_ob = normalize_ob
        if self.normalize_ob:
            self.ob_rm = np.zeros(self.ob_space.shape)
            self.ob_rv = np.ones(self.ob_space.shape)
            self.alpha = 0.001

    def update_ob_rms(self, ob):
        self.ob_rm = self.ob_rm * (1-self.alpha) + self.alpha * ob
        self.ob_rv = self.ob_rv * (1-self.alpha) + \
            self.alpha * np.square(ob-self.ob_rm)

    def prepro(self, ob):
        if self.normalize_ob:
            ob = (ob - self.ob_rm) / (np.sqrt(self.ob_rv) + 1e-8)
            ob = np.clip(ob, -5, 5)
        return ob

    def prepro_with_update(self, ob):
        if self.normalize_ob:
            self.update_ob_rms(ob)
            ob = (ob - self.ob_rm) / (np.sqrt(self.ob_rv) + 1e-8)
            ob = np.clip(ob, -5, 5)
        return ob
