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


class BasePd(object):
    """
    Base class of probablistic distribution
    """
    def sample(self, params, sample_shape):
        """
        sampling
        """
        raise NotImplementedError

    def llh(self, x, params):
        """
        log liklihood
        """
        raise NotImplementedError

    def kl_pq(self, p_params, q_params):
        """
        KL divergence between p and q
        """
        raise NotImplementedError

    def ent(self, params):
        """
        entropy
        """
        raise NotImplementedError
