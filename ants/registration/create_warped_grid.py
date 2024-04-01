

__all__ = ['create_warped_grid']

import numpy as np

from ..core import ants_image as iio
from ..core import ants_image_io as iio2
from .apply_transforms import apply_transforms

def create_grid_source(size=(250, 250),
                       sigma=(0.5, 0.5),
                       grid_spacing=(5.0, 5.0),
                       grid_offset=(0.0, 0.0),
                       spacing=(0.2, 0.2),
                       pixeltype='float'):
    pass

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
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data( 'r16' ) )
    >>> mi = ants.image_read( ants.get_ants_data( 'r64' ) )
    >>> mygr = ants.create_warped_grid( mi )
    >>> mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyN') )
    >>> mywarpedgrid = ants.create_warped_grid( mi, grid_directions=(False,True),
                            transform=mytx['fwdtransforms'], fixed_reference_image=fi )
    """
    if isinstance(image, iio.ANTsImage):
        if len(grid_directions) != image.dimension:
            grid_directions = [True]*image.dimension
        garr = image.numpy() * 0 + foreground
    else:
        if not isinstance(image, (list, tuple)):
            raise ValueError('image arg must be ANTsImage or list or tuple')
        if len(grid_directions) != len(image):
            grid_directions = [True]*len(image)
        garr = np.zeros(image) + foreground
        image = iio2.from_numpy(garr)

    idim = garr.ndim
    gridw = grid_width

    for d in range(idim):
        togrid = np.arange(-1, garr.shape[d]-1, step=grid_step)
        for i in range(len(togrid)):
            if (d == 0) & (idim == 3) & (grid_directions[d]):
                garr[togrid[i]:(togrid[i]+gridw),...] = background
                garr[0,...] = background
                garr[-1,...] = background
            if (d == 1) & (idim == 3) & (grid_directions[d]):
                garr[:,togrid[i]:(togrid[i]+gridw),:] = background
                garr[:,0,:] = background
                garr[:,-1,:] = background
            if (d == 2) & (idim == 3) & (grid_directions[d]):
                garr[...,togrid[i]:(togrid[i]+gridw)] = background
                garr[...,0] = background
                garr[...,-1] = background
            if (d == 0) & (idim == 2) & (grid_directions[d]):
                garr[togrid[i]:(togrid[i]+gridw),:] = background
                garr[0,:] = background
                garr[-1,:] = background
            if (d == 1) & (idim == 2) & (grid_directions[d]):
                garr[:,togrid[i]:(togrid[i]+gridw)] = background
                garr[:,0] = background
                garr[:,-1] = background


    gimage = image.new_image_like(garr)

    if (transform is not None) and (fixed_reference_image is not None):
        return apply_transforms( fixed=fixed_reference_image, moving=gimage,
                               transformlist=transform ) 
    else:
        return gimage