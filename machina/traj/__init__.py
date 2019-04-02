"""
- This package is the bockbone of machina.
   - :class:`Environment<machina.envs.gym_env.GymEnv>` and :py:mod:`Algorithm<machina.algos>` are seperated by this package.
- Traj class
   - :data:`epis` sampled by :py:mod:`sampler<machina.samplers>` are changed into the class :class:`Traj<machina.traj.traj.Traj>`.
   - :class:`Traj<machina.traj.traj.Traj>` is the :py:class:`list` of :py:class:`dict`.
- Iterator
   - Output is batch, a :py:class:`dict` keys of which are MDP elements such as :data:`obs`, :data:`acs`, :data:`rews`, etc.
   - Methods of :py:meth:`iterate*<machina.traj.traj.Traj.iterate>` are used for On-Policy algorithms.
   - Methods of :py:meth:`random*<machina.traj.traj.Traj.random_batch>` are used for Off-Policy algorithms.
"""
from machina.traj.traj import Traj
