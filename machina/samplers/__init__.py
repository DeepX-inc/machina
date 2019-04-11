"""
 - This package samples episodes from :class:`Environment<machina.envs.gym_env.GymEnv>` in multiprocessing.
 - Inputs are :class:`Policy<machina.pols.base.BasePol>` and maximum steps or episodes.
 - Output is :py:class:`ndarray` of :py:class:`dict` of :py:class:`list`.
"""
from machina.samplers.epi_sampler import EpiSampler
from machina.samplers.distributed_epi_sampler import DistributedEpiSampler
