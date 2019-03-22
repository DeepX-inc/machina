import pkg_resources


__version__ = pkg_resources.get_distribution('machina-rl').version


from machina import algos  # NOQA
from machina import envs  # NOQA
from machina import models  # NOQA
from machina import noise  # NOQA
from machina import optims  # NOQA
from machina import pds  # NOQA
from machina import pols  # NOQA
from machina import prepro  # NOQA
from machina import samplers  # NOQA
from machina import traj  # NOQA
from machina import vfuncs  # NOQA
