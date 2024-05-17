"""
Functions for plotting ants images
"""


__all__ = [
    "plot_directory"
]

import fnmatch
import math
import os
import warnings

from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.mlab as mlab
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import numpy as np

import ants


def plot_directory(
    directory,
    recursive=False,
    regex="*",
    save_prefix="",
    save_suffix="",
    axis=None,
    **kwargs
):
    """
    Create and save an ANTsPy plot for every image matching a given regular
    expression in a directory, optionally recursively. This is a good function
    for quick visualize exploration of all of images in a directory

    ANTsR function: N/A

    Arguments
    ---------
    directory : string
        directory in which to search for images and plot them

    recursive : boolean
        If true, this function will search through all directories under
        the given directory recursively to make plots.
        If false, this function will only create plots for images in the
        given directory

    regex : string
        regular expression used to filter out certain filenames or suffixes

    save_prefix : string
        sub-string that will be appended to the beginning of all saved plot filenames.
        Default is to add nothing.

    save_suffix : string
        sub-string that will be appended to the end of all saved plot filenames.
        Default is add nothing.

    kwargs : keyword arguments
        any additional arguments to pass onto the `ants.plot` function.
        e.g. overlay, alpha, cmap, etc. See `ants.plot` for more options.

    Example
    -------
    >>> import ants
    >>> ants.plot_directory(directory='~/desktop/testdir',
                            recursive=False, regex='*')
    """

    def has_acceptable_suffix(fname):
        suffixes = {".nii.gz"}
        return sum([fname.endswith(sx) for sx in suffixes]) > 0

    if directory.startswith("~"):
        directory = os.path.expanduser(directory)

    if not os.path.isdir(directory):
        raise ValueError("directory %s does not exist!" % directory)

    for root, dirnames, fnames in os.walk(directory):
        for fname in fnames:
            if fnmatch.fnmatch(fname, regex) and has_acceptable_suffix(fname):
                load_fname = os.path.join(root, fname)
                fname = fname.replace(".".join(fname.split(".")[1:]), "png")
                fname = fname.replace(".png", "%s.png" % save_suffix)
                fname = "%s%s" % (save_prefix, fname)
                save_fname = os.path.join(root, fname)
                img = ants.image_read(load_fname)

                if axis is None:
                    axis_range = [i for i in range(img.dimension)]
                else:
                    axis_range = axis if isinstance(axis, (list, tuple)) else [axis]

                if img.dimension > 2:
                    for axis_idx in axis_range:
                        filename = save_fname.replace(".png", "_axis%i.png" % axis_idx)
                        ncol = int(math.sqrt(img.shape[axis_idx]))
                        ants.plot(
                            img,
                            axis=axis_idx,
                            nslices=img.shape[axis_idx],
                            ncol=ncol,
                            filename=filename,
                            **kwargs
                        )
                else:
                    filename = save_fname
                    ants.plot(img, filename=filename, **kwargs)
