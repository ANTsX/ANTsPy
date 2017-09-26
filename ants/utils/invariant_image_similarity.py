
__all__ = ['invariant_image_similarity',
           'convolve_image']

import math
from tempfile import mktemp

import numpy as np
import pandas as pd

from .. import utils
from ..core import ants_image as iio


def invariant_image_similarity(image1, image2, 
                                local_search_iterations=0, metric='MI', 
                                thetas=np.linspace(0,360,5),
                                thetas2=np.linspace(0,360,5),
                                thetas3=np.linspace(0,360,5),  
                                scale_image=1, do_reflection=False,
                                txfn=None, transform='Affine'):
    """
    Similarity metrics between two images as a function of geometry

    Compute similarity metric between two images as image is rotated about its
    center w/ or w/o optimization

    ANTsR function: `invariantImageSimilarity`

    Arguments
    ---------
    image1 : ANTsImage
        reference image
    
    image2 : ANTsImage
        moving image
    
    local_search_iterations : integer
        integer controlling local search in multistart
    
    metric : string
        which metric to use
            MI
            GC
    thetas : 1D-ndarray/list/tuple
        numeric vector of search angles in degrees
    
    thetas2 : 1D-ndarray/list/tuple
        numeric vector of search angles in degrees around principal axis 2 (3D)
    
    thetas3 : 1D-ndarray/list/tuple
        numeric vector of search angles in degrees around principal axis 3 (3D)
    
    scale_image : scalar
        global scale
    
    do_reflection : boolean
        whether to reflect image about principal axis
    
    txfn : string (optional)
        if present, write optimal tx to .mat file
    
    transform : string
        type of transform to use
            Rigid
            Similarity
            Affine 

    Returns
    -------
    pd.DataFrame
        dataframe with metric values and transformation parameters

    Example
    -------
    >>> import ants
    >>> img1 = ants.image_read(ants.get_ants_data('r16'))
    >>> img2 = ants.image_read(ants.get_ants_data('r64'))
    >>> metric = ants.invariant_image_similarity(img1,img2)
    """
    if transform not in {'Rigid', 'Similarity', 'Affine'}:
        raise ValueError('transform must be one of Rigid/Similarity/Affine')

    if image1.pixeltype != 'float':
        image1 = image1.clone('float')
    if image2.pixeltype != 'float':
        image2 = image2.clone('float')

    if txfn is None:
        txfn = mktemp(suffix='.mat')

    # convert thetas to radians
    thetain = (thetas * math.pi) / 180.
    thetain2 = (thetas2 * math.pi) / 180.
    thetain3 = (thetas3 * math.pi) / 180.

    image1 = utils.iMath(image1, 'Normalize')
    image2 = utils.iMath(image2, 'Normalize')

    idim = image1.dimension
    fpname = ['FixedParam%i'%i for i in range(1,idim+1)]

    if not do_reflection:
        libfn = utils.get_lib_fn('invariantImageSimilarity_%s%iD' % (transform, idim))
        r1 = libfn(image1.pointer, 
                    image2.pointer,
                    list(thetain), 
                    list(thetain2), 
                    list(thetain3),
                    local_search_iterations, 
                    metric,
                    scale_image, 
                    int(do_reflection),
                    txfn)
        r1 = np.asarray(r1)

        pnames = ['Param%i'%i for i in range(1,r1.shape[1])]
        pnames[(len(pnames)-idim):len(pnames)] = fpname

        r1 = pd.DataFrame(r1, columns=['MetricValue']+pnames)
        return r1, txfn
    else:
        txfn1 = mktemp(suffix='.mat')
        txfn2 = mktemp(suffix='.mat')
        txfn3 = mktemp(suffix='.mat')
        txfn4 = mktemp(suffix='.mat')

        libfn = utils.get_lib_fn('invariantImageSimilarity_%s%iD' % (transform, idim))
        ## R1 ##
        r1 = libfn(image1.pointer,
                    image2.pointer,
                    list(thetain), 
                    list(thetain2), 
                    list(thetain3),
                    local_search_iterations, 
                    metric,
                    scale_image, 
                    0,
                    txfn1)
        r1 = np.asarray(r1)
        pnames = ['Param%i'%i for i in range(1,r1.shape[1])]
        pnames[(len(pnames)-idim):len(pnames)] = fpname
        r1 = pd.DataFrame(r1, columns=['MetricValue']+pnames)

        ## R2 ##
        r2 = libfn(image1.pointer,
                    image2.pointer,
                    list(thetain), 
                    list(thetain2), 
                    list(thetain3),
                    local_search_iterations, 
                    metric,
                    scale_image, 
                    1,
                    txfn2)
        r2 = np.asarray(r2)
        r2 = pd.DataFrame(r2, columns=['MetricValue']+pnames)

        ## R3 ##
        r3 = libfn(image1.pointer,
                    image2.pointer,
                    list(thetain), 
                    list(thetain2), 
                    list(thetain3),
                    local_search_iterations, 
                    metric,
                    scale_image, 
                    2,
                    txfn3)
        r3 = np.asarray(r3)
        r3 = pd.DataFrame(r3, columns=['MetricValue']+pnames)

        ## R4 ##
        r4 = libfn(image1.pointer,
                    image2.pointer,
                    list(thetain), 
                    list(thetain2), 
                    list(thetain3),
                    local_search_iterations, 
                    metric,
                    scale_image, 
                    3,
                    txfn4)
        r4 = np.asarray(r4)
        r4 = pd.DataFrame(r4, columns=['MetricValue']+pnames)

        rmins = [np.min(r1.iloc[:,0]), np.min(r2.iloc[:,0]), np.min(r3.iloc[:,0]), np.min(r4.iloc[:,0])]
        ww = np.argmin(rmins)

        if ww == 0:
            return r1, txfn1
        elif ww == 1:
            return r2, txfn2
        elif ww == 2:
            return r3, txfn3
        elif ww == 3:
            return r4, txfn4


def convolve_image(image, kernel_image, crop=True):
    """
    Convolve one image with another

    ANTsR function: `convolveImage`

    Arguments
    ---------
    image : ANTsImage
        image to convolve

    kernel_image : ANTsImage
        image acting as kernel

    crop : boolean
        whether to automatically crop kernel_image

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('r16'))
    >>> convimg = ants.make_image( (3,3), (1,0,1,0,-4,0,1,0,1) )
    >>> convout = ants.convolve_image( fi, convimg )
    >>> convimg2 = ants.make_image( (3,3), (0,1,0,1,0,-1,0,-1,0) )
    >>> convout2 = ants.convolve_image( fi, convimg2 )
    """
    if not isinstance(image, iio.ANTsImage):
        raise ValueError('image must be ANTsImage type')
    if not isinstance(kernel_image, iio.ANTsImage):
        raise ValueError('kernel must be ANTsImage type')
    
    orig_ptype = image.pixeltype
    if image.pixeltype != 'float':
        image = image.clone('float')
    if kernel_image.pixeltype != 'float':
        kernel_image = kernel_image.clone('float')

    if crop:
        kernel_image_mask = utils.get_mask(kernel_image)
        kernel_image = utils.crop_image(kernel_image, kernel_image_mask)
        kernel_image_mask = utils.crop_image(kernel_image_mask, kernel_image_mask)
        kernel_image[kernel_image_mask==0] = kernel_image[kernel_image_mask==1].mean()

    libfn = utils.get_lib_fn('convolveImageF%i' % image.dimension)
    conv_itk_image = libfn(image.pointer, kernel_image.pointer)
    conv_ants_image = iio.ANTsImage(pixeltype=image.pixeltype, dimension=image.dimension,
                                    components=image.components, pointer=conv_itk_image)

    if orig_ptype != 'float':
        conv_ants_image = conv_ants_image.clone(orig_ptype)

    return conv_ants_image



