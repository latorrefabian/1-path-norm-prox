import matplotlib
import numpy as np
import platform

if platform.system() == 'Darwin':  # OSX
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


MARKERS = ['+', 'o', '*', 'P', 's', '^', 'x']

LINE_PARAMS = {
    'markersize': 7,
    'markevery': 0.3,
    'linewidth': 1.3,
    'markerfacecolor': 'none',
    'markeredgewidth': 1.6,
    }

plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'mathtext.cal': 'serif',
    'mathtext.bf': 'serif:bold',
    'mathtext.it': 'serif:italic',
    'mathtext.sf': 'sans\\-serif',
    'font.size': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'axes.grid': False,
    'grid.linestyle': ':',
    'grid.linewidth': .8,
    'grid.color': 'lightgray',
    })

FIGURE_SIZE = {
    'icml': {
        'one_column': (2.66, 1.88),
        },
    'neurips': {
        'one_third': (2., 1.5),
        'one_half': (2.66, 1.8),
        'one_half2': (5, 5)
    }
}


def lineplot(
        variables, xlabel, ylabel, xscale, yscale,
        conference, size, filename, xlim=None, ylim=None):
    """
    Args:
        variables (dict): if it is a dictionary with numpy arrays as values,
            then it will use keys as labels and values (tuple of either one or
            two np.arrays) for a line plot. If it is a dictionary composed of
            dictionaries (each of which has the characteristics previously
            described), then it will plot in the same row, the figures
            corresponding to each value in the dictionary. Each figure will
            have as title the corresponding key in the dictionary.

            When multiple dictionaries are passed as inputs, the scale and
            range of the axes will be shared, and only the right-most figure
            will include a legend (if legend=True). The intended use is to plot
            different level curves of the same function.

    """

    figsize = FIGURE_SIZE[conference][size]

    if type(next(iter(variables.values()))) is dict:
        if len(variables) == 1:
            variables = next(iter(variables.values()))
        else:
            _lineplots(
                    variables, xlabel=xlabel, ylabel=ylabel,
                    figsize=figsize, xscale=xscale)
    else:
        _lineplot(
                variables, xlabel=xlabel, ylabel=ylabel,
                figsize=figsize, xscale=xscale)

    plt.xscale(xscale)
    plt.yscale(yscale)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout(pad=0.15)
    plt.savefig(filename, bbox='tight', format='pdf')
    plt.clf()


def _add_plot(ax, i, value, label, xscale):
    """
    Add line plot to axis

    Args:
        ax (matplotlib.Axes): axis to add the plot
        i (int): index of the plot (to change marker)
        value (tuple): either 'x' and 'y' values, or only 'y' values
        label (str): text for the legend
        xscale (str): scale for the horizontal axis

    """
    if len(value) == 1:  # only 'y' values
        y = value[0]
        if xscale == 'log':
            x = np.arange(1, len(y) + 1, dtype=float)
        else:
            x = np.arange(len(y), dtype=float)
    elif len(value) == 2:  # both 'x' and 'y' values
        x, y = value
    else:
        raise ValueError(
                '"variables" should be a tuple of length one or two')

    ax.plot(
            x, y, label=label, marker=MARKERS[i % len(MARKERS)], **LINE_PARAMS)


def _lineplot(variables, xlabel, ylabel, figsize, xscale):
    """
    Overlayed line plots with legend, single figure.

    Args:
        variables (dict): key, value pairs where the key will be used as the
            legend text, and the value should be a tuple of either one or two
            arrays. If it is only one array then it will be used for the 'y'
            coordinates, and the 'x' coordinates will be an array of contiguous
            integers starting from 1 (if the x-axis has a log scale) or 0
            (else). If the value consists of two arrays, the first will be used
            for the x-axis values, and the second for the y-axis values.
        xlabel (str): text for the horizontal axis
        ylabel (str): text for the vertical axis
        figsize (tuple): size of the figure
        xscale (str): scale for the horizontal axis

    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, (key, value) in enumerate(variables.items()):
        _add_plot(ax=ax, i=i, value=value, label=key, xscale=xscale)

    ax.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def _lineplots(variables, xlabel, ylabel, figsize, xscale):
    """
    Overlayed line plots with legend, multiple figures in one row.

    Args:
        variables (dict): key, value pairs where the key will be used as the
            title (top) of the figure, and the value should be a dictionary
            which is passed to the _iterplot method.
        xlabel (str): text for the horizontal axis
        ylabel (str): text for the vertical axis
        figsize (tuple): size of the figure
        xscale (str): scale for the horizontal axis

    """
    fig, axs = plt.subplots(
            figsize=figsize, nrows=1,
            ncols=len(variables), sharex=True, sharey=True)
    for i, (key, value) in enumerate(variables.items()):
        for j, (key_, value_) in enumerate(value.items()):
            _add_plot(ax=axs[i], i=j, value=value_, label=key_, xscale=xscale)
        axs[i].set_title(key)
        axs[i].grid(True)

        if i == len(variables) - 1:
            plt.legend()
        if i == 0:
            axs[i].set_ylabel(ylabel)

