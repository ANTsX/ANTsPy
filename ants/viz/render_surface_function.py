__all__ = ["render_surface_function"]

import os
import warnings
import skimage.measure
import webcolors as wc
from ..registration import resample_image

try:
    import chart_studio.plotly as py
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    from plotly.graph_objs import *
    from plotly import figure_factory as FF
except:
    warnings.warn(
        "Cant import Plotly. Install it `pip install chart_studio` if you want to use ants.render_surface_function"
    )


def render_surface_function(
    surfimg,
    funcimg=None,
    alphasurf=0.2,
    alphafunc=1.0,
    isosurf=0.5,
    isofunc=0.5,
    smoothsurf=None,
    smoothfunc=None,
    cmapsurf="grey",
    cmapfunc="red",
    filename=None,
    notebook=False,
    auto_open=False,
):
    """
    Render an image as a base surface and an optional collection of other image.

    ANTsR function: `renderSurfaceFunction`
        NOTE: The ANTsPy version of this function is actually completely different
        than the ANTsR version, although they should produce similar results.

    Arguments
    ---------
    surfimg : ANTsImage
        Input image to use as rendering substrate.

    funcimg : ANTsImage
        Input list of images to use as functional overlays.

    alphasurf : scalar
        alpha for the surface contour

    alphafunc : scalar
        alpha value for functional blobs

    isosurf : scalar
        intensity level that defines lower threshold for surface image

    isofunc : scalar
        intensity level that defines lower threshold for functional image

    smoothsurf  : scalar (optional)
        smoothing for the surface image

    smoothfunc : scalar (optional)
        smoothing for the functional image

    cmapsurf : string
        color map for surface image

    cmapfunc : string
        color map for functional image

    filename : string
        where to save rendering. if None, will plot interactively

    notebook : boolean
        whether you're in a jupyter notebook.

    Returns
    -------
    N/A

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_ants_data('mni'))
    >>> mnia = ants.image_read(ants.get_ants_data('mnia'))
    >>> ants.render_surface_function(mni, mnia, alphasurf=0.1, filename='/users/ncullen/desktop/surffnc.png')
    """
    cmap_dict = {
        "grey": "Greys",
        "gray": "Greys",
        "red": "Reds",
        "green": "Greens",
        "jet": "Jet",
    }
    if surfimg.dimension != 3:
        raise ValueError("surfimg must be 3D")

    # if (filename is None) and (not notebook_render):
    #    raise Exception('Must either 1) give filename, 2) set `html_render`=True or 3) set `notebook_render`=True')

    if notebook:
        init_notebook_mode(connected=True)

    fig_list = []
    fig_data_list = []

    surfimg = resample_image(surfimg, (3, 3, 3))
    surfimg_arr = surfimg.numpy()
    surfverts, surffaces, _, _ = skimage.measure.marching_cubes_lewiner(
        surfimg_arr, isosurf, spacing=(1, 1, 1)
    )

    surffig = FF.create_trisurf(
        x=surfverts[:, 0],
        y=surfverts[:, 1],
        z=surfverts[:, 2],
        colormap=cmap_dict.get(cmapsurf, cmapsurf),
        plot_edges=False,
        simplices=surffaces,
    )

    surffig["data"][0].update(opacity=alphasurf)

    fig_list.append(surffig)
    fig_data_list.append(surffig.data[0])

    if funcimg is not None:
        if not isinstance(funcimg, (tuple, list)):
            funcimg = [funcimg]
        if not isinstance(alphafunc, (tuple, list)):
            alphafunc = [alphafunc] * len(funcimg)
        if not isinstance(isofunc, (tuple, list)):
            isofunc = [isofunc] * len(funcimg)
        if not isinstance(cmapfunc, (tuple, list)):
            cmapfunc = [cmapfunc] * len(funcimg)
            # cmapfunc = [cmap_dict.get(c,c) for c in cmapfunc]

        for i in range(len(cmapfunc)):
            cmapfunc[i] = "rgb%s" % str(wc.name_to_rgb(cmapfunc[i]))
            cmapfunc[i] = [cmapfunc[i]] * 2

        for func_idx, fimg in enumerate(funcimg):
            if fimg.dimension != 3:
                raise ValueError("all funcimgs must be 3D")

            fimg = resample_image(fimg, (3, 3, 3))
            funcimg_arr = fimg.numpy()
            funcverts, funcfaces, _, _ = skimage.measure.marching_cubes_lewiner(
                funcimg_arr, isofunc[func_idx], spacing=(1, 1, 1)
            )
            funcfig = FF.create_trisurf(
                x=funcverts[:, 0],
                y=funcverts[:, 1],
                z=funcverts[:, 2],
                plot_edges=False,
                simplices=funcfaces,
                colormap=cmapfunc[func_idx],
            )
            funcfig["data"][0].update(opacity=alphafunc[func_idx])
            fig_list.append(funcfig)
            fig_data_list.append(funcfig.data[0])

    if filename is not None:
        save_file = "png"
        image_filename = filename
        filename = image_filename.split(".")[0] + ".html"
    else:
        image_filename = "ants_plot"
        filename = "ants_plot.html"
        save_file = None

    try:
        plot(
            fig_data_list,
            image=save_file,
            filename=filename,
            image_filename=image_filename,
            auto_open=auto_open,
        )
    except PermissionError:
        print(
            "PermissionError caught - are you running jupyter console? Try launching it with sudo privledges (e.g. `sudo jupyter-console`)"
        )
