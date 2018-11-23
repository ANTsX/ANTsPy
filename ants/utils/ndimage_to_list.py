__all__ = ['ndimage_to_list',
           'list_to_ndimage']

import numpy as np
# import itk as itk
from ..core import ants_image as iio
from ..core import ants_image_io as iio2
from .. import utils


def list_to_ndimage( image, image_list ):
    """
    Merge list of multiple scalar ANTsImage types of dimension into one
    ANTsImage of dimension plus one

    ANTsR function: `mergeListToNDImage`

    Arguments
    ---------
    image : target image space
    image_list : list/tuple of ANTsImage types
        scalar images to merge into target image space

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> image2 = ants.image_read(ants.get_ants_data('r16'))
    >>> imageTar = ants.make_image( ( *image2.shape, 2 ) )
    >>> image3 = ants.list_to_ndimage( imageTar, [image,image2])
    >>> image3.dimension == 3
    """
    inpixeltype = image_list[0].pixeltype
    dimension = image_list[0].dimension
    components = len(image_list)

    for imageL in image_list:
        if not isinstance(imageL, iio.ANTsImage):
            raise ValueError('list may only contain ANTsImage objects')
        if image.pixeltype != inpixeltype:
            raise ValueError('all images must have the same pixeltype')

    dimensionout = ( *image_list[0].shape, len( image_list )  )
    newImage = iio2.make_image(
      dimensionout,
      spacing = iio.get_spacing( image ),
      origin = iio.get_origin( image ),
      direction = iio.get_direction( image ),
      pixeltype = inpixeltype
      )
    # FIXME - should implement paste image filter from ITK
    for x in range( len( image_list ) ):
        if dimension == 2:
            newImage[:,:,x] = image_list[x][:,:]
        if dimension == 3:
            newImage[:,:,:,x] = image_list[x][:,:,:]
    return newImage





def ndimage_to_list(image):
    """
    Split a n dimensional ANTsImage into a list
    of n-1 dimensional ANTsImages

    Arguments
    ---------
    image : ANTsImage
        n-dimensional image to split

    Returns
    -------
    list of ANTsImage types

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> image2 = ants.image_read(ants.get_ants_data('r16'))
    >>> imageTar = ants.make_image( ( *image2.shape, 2 ) )
    >>> image3 = ants.list_to_ndimage( imageTar, [image,image2])
    >>> image3.dimension == 3
    >>> images_unmerged = ants.ndimage_to_list( image3 )
    >>> len(images_unmerged) == 2
    >>> images_unmerged[0].dimension == 2
    """
    inpixeltype = image.pixeltype
    dimension = image.dimension
    components = 1
    imageShape = image.shape
    nSections = imageShape[ dimension - 1 ]
    subdimension = dimension - 1
    suborigin = iio.get_origin( image )[0:subdimension]
    subspacing = iio.get_spacing( image )[0:subdimension]
    subdirection = np.eye( subdimension )
    for i in range( subdimension ):
        subdirection[i,:] = iio.get_direction( image )[i,0:subdimension]
    subdim = image.shape[ 0:subdimension ]
    imagelist = []
    for i in range( nSections ):
        img = utils.slice_image( image, axis = subdimension, idx = i )
        iio.set_spacing( img, subspacing )
        iio.set_origin( img, suborigin )
        iio.set_direction( img, subdirection )
        imagelist.append( img )

    return imagelist
