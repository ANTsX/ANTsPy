"""
Create a static 2D image of a 2D ANTsImage
or a tile of slices from a 3D ANTsImage
"""


__all__ = ['plot']

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_stable(image, cmap='Greys_r', axis=0, nslices=12, slices=None, ncol=4):
    """
    Plot an ANTsImage
    
    ANTsR function: `plot`

    Immediate Goals:
        X support 2D images
        X support 3D images with `nslices` and `axis` arguments
        - support single overlay
        
    Future Goals:
        - support multiple overlays
        - support multiple colorschemes

    plot(x, y, color.img = "white", color.overlay = c("jet",
          "red", "blue", "green", "yellow", "viridis", "magma", "plasma", "inferno"),
          axis = 2, slices, colorbar = missing(y), title.colorbar, title.img,
          title.line = NA, color.colorbar, window.img, window.overlay, quality = 2,
          outname = NA, alpha = 1, newwindow = FALSE, nslices = 10,
          domainImageMap = NA, ncol = 4, useAbsoluteScale = FALSE,
          doCropping = TRUE, ...)

    if overlay is not None:
        if overlay.dimension != image.dimension:
            raise ValueError('image and overlay(s) must have same dimension')
        overlay_arr = overlay.numpy()
        overlay_arr[overlay_arr <= 0] = None

    Example
    -------
    >>> ## 2D images ##
    >>> import ants
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> ants.plot(img)
    >>> ## 3D images ##
    >>> import ants
    >>> img3d = ants.image_read(ants.get_data('ch2'))
    >>> ants.plot(img3d)
    >>> ants.plot(img3d, axis=0, nslices=5) # slice numbers
    >>> ants.plot(img3d, axis=1, nslices=5) # different axis
    >>> ants.plot(img3d, axis=2, nslices=5) # different slices
    >>> ants.plot(img3d, nslices=1) # one slice
    >>> ants.plot(img3d, slices=(50,70,90)) # absolute slices
    >>> ants.plot(img3d, slices=(0.4,0.6,0.8)) # relative slices
    >>> ants.plot(img3d, slices=50) # one absolute slice
    >>> ants.plot(img3d, slices=0.6) # one relative slice
    """
    # Plot 2D image
    if image.dimension == 2:
        img_arr = image.numpy()
        fig, ax = plt.subplots()

        ax.imshow(img_arr, cmap=cmap)
        plt.axis('off')
        plt.show()

    # Plot 3D image
    elif image.dimension == 3:
        img_arr = image.numpy()

        # reorder dims so that chosen axis is first
        img_arr = np.rollaxis(img_arr, axis)

        if slices is None:
            nonzero = np.where(np.abs(img_arr)>0)[0]
            min_idx = nonzero[0]
            max_idx = nonzero[-1]
            slice_idxs = np.linspace(min_idx+1, max_idx-1, nslices).astype('int')
        else:
            if isinstance(slices, (int,float)):
                slices = [slices]
            if slices[0] < 1:
                slices = [int(s*img_arr.shape[0]) for s in slices]
            slice_idxs = slices
            nslices = len(slices)

        # only have one row if nslices <= 6 and user didnt specify ncol
        if (nslices <= 6) and (ncol==4):
            ncol = nslices

        # calculate grid size
        nrow = math.ceil(nslices / ncol)

        xdim = img_arr.shape[1]
        ydim = img_arr.shape[2]

        fig = plt.figure(figsize=((ncol+1)*1.5*(ydim/xdim), (nrow+1)*1.5)) 

        gs = gridspec.GridSpec(nrow, ncol,
                 wspace=0.0, hspace=0.0, 
                 top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                 left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

        slice_idx_idx = 0
        for i in range(nrow):
            for j in range(ncol):
                if slice_idx_idx < len(slice_idxs):
                    im = img_arr[slice_idxs[slice_idx_idx]]
                    ax = plt.subplot(gs[i,j])
                    ax.imshow(im, cmap=cmap)
                    ax.axis('off')
                    slice_idx_idx += 1

        plt.show()



def plot(image, overlay=None, cmap='Greys_r', overlay_cmap='jet', axis=0, nslices=12, slices=None, ncol=4):
    """
    Plot an ANTsImage
    
    ANTsR function: `plot`

    Immediate Goals:
        X support 2D images
        X support 3D images with `nslices` and `axis` arguments
        - support single overlay
        
    Future Goals:
        - support multiple overlays
        - support multiple colorschemes

    plot(x, y, color.img = "white", color.overlay = c("jet",
          "red", "blue", "green", "yellow", "viridis", "magma", "plasma", "inferno"),
          axis = 2, slices, colorbar = missing(y), title.colorbar, title.img,
          title.line = NA, color.colorbar, window.img, window.overlay, quality = 2,
          outname = NA, alpha = 1, newwindow = FALSE, nslices = 10,
          domainImageMap = NA, ncol = 4, useAbsoluteScale = FALSE,
          doCropping = TRUE, ...)

    if overlay is not None:
        if overlay.dimension != image.dimension:
            raise ValueError('image and overlay(s) must have same dimension')
        overlay_arr = overlay.numpy()
        overlay_arr[overlay_arr <= 0] = None

    Example
    -------
    >>> ## 2D images ##
    >>> import ants
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> ants.plot(img)
    >>> overlay = (img.kmeans_segmentation(k=3)['segmentation']==3)*(img.clone())
    >>> ants.plot(img, overlay)
    >>> ## 3D images ##
    >>> import ants
    >>> img3d = ants.image_read(ants.get_data('ch2'))
    >>> ants.plot(img3d)
    >>> ants.plot(img3d, axis=0, nslices=5) # slice numbers
    >>> ants.plot(img3d, axis=1, nslices=5) # different axis
    >>> ants.plot(img3d, axis=2, nslices=5) # different slices
    >>> ants.plot(img3d, nslices=1) # one slice
    >>> ants.plot(img3d, slices=(50,70,90)) # absolute slices
    >>> ants.plot(img3d, slices=(0.4,0.6,0.8)) # relative slices
    >>> ants.plot(img3d, slices=50) # one absolute slice
    >>> ants.plot(img3d, slices=0.6) # one relative slice
    >>> ## Overlay Example ##
    >>> import ants
    >>> img = ants.image_read(ants.get_data('ch2'))
    >>> overlay = img.clone()
    >>> overlay = overlay*(overlay>105.)
    >>> ants.plot(img, overlay)
    """
    # Plot 2D image
    if image.dimension == 2:
        img_arr = image.numpy()

        if overlay is not None:
            ov_arr = overlay.numpy()
            ov_arr[np.abs(ov_arr) == 0] = np.nan

        fig, ax = plt.subplots()

        ax.imshow(img_arr, cmap=cmap)

        if overlay is not None:
            ax.imshow(ov_arr, cmap=overlay_cmap)

        plt.axis('off')
        plt.show()

    # Plot 3D image
    elif image.dimension == 3:
        img_arr = image.numpy()

        if overlay is not None:
            ov_arr = overlay.numpy()
            ov_arr[np.abs(ov_arr) == 0] = np.nan

        # reorder dims so that chosen axis is first
        img_arr = np.rollaxis(img_arr, axis)

        if slices is None:
            nonzero = np.where(np.abs(img_arr)>0)[0]
            min_idx = nonzero[0]
            max_idx = nonzero[-1]
            slice_idxs = np.linspace(min_idx+1, max_idx-1, nslices).astype('int')
        else:
            if isinstance(slices, (int,float)):
                slices = [slices]
            if slices[0] < 1:
                slices = [int(s*img_arr.shape[0]) for s in slices]
            slice_idxs = slices
            nslices = len(slices)

        # only have one row if nslices <= 6 and user didnt specify ncol
        if (nslices <= 6) and (ncol==4):
            ncol = nslices

        # calculate grid size
        nrow = math.ceil(nslices / ncol)

        xdim = img_arr.shape[1]
        ydim = img_arr.shape[2]

        fig = plt.figure(figsize=((ncol+1)*1.5*(ydim/xdim), (nrow+1)*1.5)) 

        gs = gridspec.GridSpec(nrow, ncol,
                 wspace=0.0, hspace=0.0, 
                 top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                 left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

        slice_idx_idx = 0
        for i in range(nrow):
            for j in range(ncol):
                if slice_idx_idx < len(slice_idxs):
                    im = img_arr[slice_idxs[slice_idx_idx]]
                    ax = plt.subplot(gs[i,j])
                    ax.imshow(im, cmap=cmap)
                    if overlay is not None:
                        ov = ov_arr[slice_idxs[slice_idx_idx]]
                        ax.imshow(ov, alpha=0.9, cmap=overlay_cmap)
                    ax.axis('off')
                    slice_idx_idx += 1

        plt.show()

