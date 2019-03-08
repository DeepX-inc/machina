import pkg_resources


__version__ = pkg_resources.get_distribution('machina').version


from machina.traj import epi_functional
