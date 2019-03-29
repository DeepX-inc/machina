import numpy as np
from collections import OrderedDict

def flatten_to_dict(flatten_obs, dict_space, dict_keys=None):
    if dict_keys is None:
        dict_keys = dict_space.spaces.keys()
    obs_dict = OrderedDict()
    begin_index = 0
    end_index = 0
    for key in dict_keys:
        origin_shape = dict_space.spaces[key].shape
        end_index += np.prod(origin_shape)
        dim = len(flatten_obs.shape)
        if dim == 1:
            obs_dict[key] = flatten_obs[begin_index:end_index].reshape(origin_shape)
        else:
            obs_dict[key] = flatten_obs[:, begin_index:end_index].reshape((-1,) + origin_shape)
        begin_index = end_index
    return obs_dict
