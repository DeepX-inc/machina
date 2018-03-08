from cached_property import cached_property
import os
import warnings

import pandas as pd

try:
    import matplotlib  # NOQA

    _available = True

except (ImportError, TypeError):
    _available = False


@cached_property
def _check_available():
    if not _available:
        warnings.warn('matplotlib is not installed on your environment, '
                      'so nothing will be plotted at this time. '
                      'Please install matplotlib to plot figures.\n\n'
                      '  $ pip install matplotlib\n')
    return _available


def plot_scores(filename, key, x_key, label=None, title=None,
                color=None, result_directory=None,
                plot_legend=True, xlim=None, ylim=None,
                y_label=None):
    if _check_available:
        import matplotlib.pyplot as plt
    else:
        return

    if label is None:
        label = key

    f = plt.figure()
    a = f.add_subplot(111)
    scores = pd.read_csv(filename)
    x = scores[x_key]
    mean = scores[key + 'Average']
    std = scores[key + 'Std']
    a.plot(x, mean, label=label)
    a.fill_between(x,
                   mean - std,
                   mean + std,
                   color=color,
                   alpha=0.2)

    a.set_xlabel('steps')
    if y_label is not None:
        a.set_ylabel(y_label)
    if plot_legend is True:
        a.legend(loc='best')
    if title is not None:
        a.set_title(title)
    if xlim is not None:
        a.set_xlim(xlim)
    if ylim is not None:
        a.set_ylim(ylim)

    if result_directory is not None:
        fig_fname = os.path.join(result_directory,
                                 "{}.png".format(label))
    else:
        fig_fname = os.path.join(os.path.dirname(filename),
                                 "{}.png".format(label))
    f.savefig(fig_fname)
    plt.close()
    return fig_fname
