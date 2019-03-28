"""
- This package trains :class:`Policy<machina.pols.base.BasePol>`, :class:`V function<machina.vfuncs.state_action_vfuncs.base.BaseSAVfunc>`, :class:`Q function<machina.vfuncs.state_vfuncs.base.BaseSVfunc>`, etc. by using :py:mod:`loss_functional<machina.loss_functional>`.
- It is determined here which :py:mod:`loss_functional<machina.loss_functional>`, :py:meth:`iterater<machina.traj.traj.Traj.iterate>` are used.
- Also, It is determined how `Policy<machina.pols.base.BasePol>`, :class:`V function<machina.vfuncs.state_action_vfuncs.base.BaseSAVfunc>`, :class:`Q function<machina.vfuncs.state_vfuncs.base.BaseSVfunc>`, etc. are updated.
"""
from machina.algos import airl  # NOQA
from machina.algos import behavior_clone  # NOQA
from machina.algos import ddpg  # NOQA
from machina.algos import diayn  # NOQA
from machina.algos import diayn_sac  # NOQA
from machina.algos import gail  # NOQA
from machina.algos import mpc  # NOQA
from machina.algos import on_pol_teacher_distill  # NOQA
from machina.algos import ppo_clip  # NOQA
from machina.algos import ppo_kl  # NOQA
from machina.algos import prioritized_ddpg  # NOQA
from machina.algos import qtopt  # NOQA
from machina.algos import r2d2_sac  # NOQA
from machina.algos import sac  # NOQA
from machina.algos import svg  # NOQA
from machina.algos import trpo  # NOQA
from machina.algos import vpg  # NOQA
