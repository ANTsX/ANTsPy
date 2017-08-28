

__all__ = ['create_warped_grid']

import numpy as np

from .. import lib
from .apply_transforms import apply_transforms

def create_warped_grid(img, grid_step=10, grid_width=2, grid_directions=(True, True),
                       fixed_reference_image=None, transform=None, foreground=1, background=0):

    if len(grid_directions) != img.dimension:
        grid_directions = [True]*img.dimension

    garr = img.numpy() * 0 + foreground
    gridw = grid_width

    for d in range(img.dimension):
        togrid = np.arange(0, garr.shape[d]-grid_width, by=grid_step)
        for i in range(togrid):
            if (d == 1) & (img.dimension == 3) & (grid_directions[d]):
                garr[togrid[i]:(togrid[i]+gridw),...] = background
            if (d == 2) & (img.dimension == 3) & (grid_directions[d]):
                garr[:,togrid[i]:(togrid[i]+gridw),:] = background
            if (d == 3) & (img.dimension == 3) & (grid_directions[d]):
                garr[...,togrid[i]:(togrid[i]+gridw)] = background
            if (d == 1) & (img.dimension == 2) & (grid_directions[d]):
                garr[togrid[i]:(togrid[i]+gridw),:] = background
            if (d == 2) & (img.dimension == 2) & (grid_directions[d]):
                garr[:,togrid[i]:(togrid[i]+gridw)] = background


    gimg = img.new_image_like(garr)

    if (transform is not None) and (fixed_reference_image is not None):
        apply_transforms( fixed=fixed_reference_image, moving=gimg,
                               transformlist=transform ) 
    else:
        return gimg