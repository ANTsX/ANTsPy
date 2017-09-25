

__all__ = ['create_warped_grid']

import numpy as np

from .apply_transforms import apply_transforms

def create_warped_grid(image, grid_step=10, grid_width=2, grid_directions=(True, True),
                       fixed_reference_image=None, transform=None, foreground=1, background=0):
    """
    Deforming a grid is a helpful way to visualize a deformation field. 
    This function enables a user to define the grid parameters 
    and apply a deformable map to that grid.

    ANTsR function: `createWarpedGrid`
    
    Arguments
    ---------
    image : ANTsImage
        input image
    
    grid_step : scalar   
        width of grid blocks
    
    grid_width : scalar
        width of grid lines
    
    grid_directions : tuple of booleans 
        directions in which to draw grid lines, boolean vector
    
    fixed_reference_image : ANTsImage (optional)
        reference image space
    
    transform : list/tuple of strings (optional)
        vector of transforms
    
    foreground : scalar
        intensity value for grid blocks
    
    background : scalar
        intensity value for grid lines

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> fi = ants.image_read( ants.get_ants_data( 'r16' ) )
    >>> mi = ants.image_read( ants.get_ants_data( 'r64' ) )
    >>> mygr = ants.create_warped_grid( mi )
    >>> mytx <- ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyN') )
    >>> mywarpedgrid = ants.create_warped_grid( mi, grid_directions=(False,True),
                            transform=mytx['fwdtransforms'], fixed_reference_image=fi )
    """
    if len(grid_directions) != image.dimension:
        grid_directions = [True]*image.dimension

    garr = image.numpy() * 0 + foreground
    gridw = grid_width

    for d in range(image.dimension):
        togrid = np.arange(0, garr.shape[d]-grid_width, by=grid_step)
        for i in range(togrid):
            if (d == 1) & (image.dimension == 3) & (grid_directions[d]):
                garr[togrid[i]:(togrid[i]+gridw),...] = background
            if (d == 2) & (image.dimension == 3) & (grid_directions[d]):
                garr[:,togrid[i]:(togrid[i]+gridw),:] = background
            if (d == 3) & (image.dimension == 3) & (grid_directions[d]):
                garr[...,togrid[i]:(togrid[i]+gridw)] = background
            if (d == 1) & (image.dimension == 2) & (grid_directions[d]):
                garr[togrid[i]:(togrid[i]+gridw),:] = background
            if (d == 2) & (image.dimension == 2) & (grid_directions[d]):
                garr[:,togrid[i]:(togrid[i]+gridw)] = background


    gimage = image.new_image_like(garr)

    if (transform is not None) and (fixed_reference_image is not None):
        apply_transforms( fixed=fixed_reference_image, moving=gimage,
                               transformlist=transform ) 
    else:
        return gimage