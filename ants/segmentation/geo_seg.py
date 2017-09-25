

__all__ = ['geo_seg']

import math

from .. import utils
from .. import registration as reg
from ..core import ants_image as iio
from ..core import ants_image_io as iio2

from .kmeans import kmeans_segmentation
from .atropos import atropos

def geo_seg(image, brainmask, priors, seginit=None, vesselopt=None, vesselk=2,
            grad_step=1.25, mrfval=0.1, atroposits=10, jacw=None, beta=0.9 ):
    """
    Brain segmentation based on geometric priors.
    Uses topological constraints to enhance accuracy of brain segmentation

    Arguments
    ---------
    image : ANTsImage or list/tuple of ANTsImage types
        input image or list of images (multiple features) where 1st image 
        would typically be the primary constrast
    
    brainmask : ANTsImage
        binary image
    
    priors : list/tuple ANTsImage types
        spatial priors, assume first is csf, second is gm, third is wm
    
    seginit : list/tuple of ANTsImage types
        a previously computed segmentation which should have the structure of 
        `atropos` or `kmeans_segmentation` output
    
    vesselopt : string
        one of bright, dark or none
    
    vesselk : integer
        integer for kmeans vessel-based processing
    
    grad_step : scalar
        scalar for registration
    
    mrfval : scalar
        e.g. 0.05 or 0.1
    
    atroposits : integer
        e.g. 5 iterations
    
    jacw : ANTsImage
        precomputed diffeo jacobian
    
    beta : scalar
        for sigma transformation ( thksig output variable )
    
    Returns
    -------
    list of ANTsImages
        segmentation result images

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('simple'))
    >>> image = ants.n3_bias_field_correction(image, 4)
    >>> image = ants.n3_bias_field_correction(image, 2)
    >>> bmk = ants.get_mask(image)
    >>> segs = ants.kmeans_segmentation(image, 3, bmk)
    >>> priors = segs['probabilityimages']
    >>> seg = ants.geo_seg(image, bmk, priors)
    """
    if isinstance(image, iio.ANTsImage):
        image = [image]

    if vesselopt is None:
        vesselopt = 'none'

    idim = image[0].dimension
    mrfterm = '[%s,%s]' % (str(mrfval), 'x'.join(['1']*idim))
    atroposits = '[%s,0]' % (str(atroposits))

    # 1 vessels via bright / dark
    if (vesselopt == 'bright') or (vesselopt == 'dark'):
        vseg = kmeans_segmentation(image[0], vesselk, brainmask)
        if vesselopt == 'bright':
            mask = utils.threshold_image(vseg['segmentation'], 1, vesselk-1)
        if vesselopt == 'dark':
            mask = utils.threshold_image(vseg['segmentation'], 2, vesselk)
    else:
        mask = brainmask.clone()

    # 2 wm /gm use topology to modify wm
    if seginit is None:
        seginit = atropos(d=idim, a=image, m=mrfterm, priorweight=0.25,
                            c=atroposits, i=priors, x=mask)

    wm = utils.threshold_image(seginit['segmentation'], 3, 3)
    wm = wm.iMath('GetLargestComponent').iMath('MD',1)
    wmp = wm * seginit['probabilityimages'][2]
    cort = utils.threshold_image(seginit['segmentation'], 2, 2)
    gm = seginit['probabilityimages'][1]
    gmp = gm + wmp

    # 3 wm / gm use diffeo to estimate gm
    if jacw is None:
        tvreg = reg.registration(gmp, wmp, type_of_transform='TVMSQ',
                                grad_step=grad_step, mask=cort)

    # 4 wm / gm / csf priors from jacobian
    # jac    = createJacobianDeterminantImage( wmp, tvreg$fwdtransforms[[1]], 0)
    jacinv = reg.create_jacobian_determinant_image( wmp, tvreg['invtransforms'][0], 0)
    jacw = reg.apply_transforms(fixed=wmp, moving=jacinv, transformlist=tvreg['fwdtransforms'])

    thkj = jacw * gm

    thksig = thkj.clone()
    # sigmoid transformation below ... 1 / ( 1 + exp(  - w / a ) )
    # where w = thkj - beta ...
    smv = 0
    a = 0.05
    thksig[ mask == 1 ] = 1.0 / ( 1 + math.exp( -1.0 * ( thkj[mask==1] - beta ) / a ) )

    seginit['probabilityimages'][1] = priors[0] + thksig
    seginit['probabilityimages'][1][seginit['probabilityimages'][1] > 1] = 1
    seginit['probabilityimages'][1] = seginit['probabilityimages'][1] * thksig.smooth_image( smv )

    # csf topology constraint based on gm/wm jacobian
    if len(priors) > 3:
        thkcsf = utils.iMath( thksig, "Neg" ) * utils.iMath( wm, "Neg" ) * utils.iMath( priors[3], "Neg" )
    if (len(priors) <= 3):
        thkcsf = utils.iMath( thksig, "Neg" ) * utils.iMath( wm, "Neg" )
    thkcsf = utils.smooth_image( thkcsf, 0.1 )
    temp = priors[0] + thkcsf
    temp[ temp > 1 ] = 1

    seginit['probabilityimages'][0] = temp.smooth_image( smv )

    # wm topology constraint based on largest connected component
    # and excluding high gm-prob voxels
    seginit['probabilityimages'][2] = priors[2] * wm.smooth_image( smv )
    seginit['probabilityimages'][2] = priors[2] * utils.iMath( thksig, "Neg")

    #
    if len(seginit['probabilityimages']) > 3:
        seginit['probabilityimages'][3] = seginit['probabilityimages'][3] * \
          utils.threhold_image(seginit['probabilityimages'][3], 0.25, 1e15 )
    #
    # now let's renormalize the priors
    # imageListToMatrix vs imagesToMatrix
    modmat = iio2.images_to_matrix( seginit['probabilityimages'], mask )
    modmatsums = modmat.sum(axis=0) # CHECK THIS => colSums( modmat )

    for k in range(modmat.shape[1]):
        modmat[:,k] = modmat[:,k] / modmatsums[k]
    seginit['probabilityimages'] = iio2.matrix_to_images(modmat, mask)

    #
    # now resegment with topology-modified priors
    s3 = atropos(d=idim, a=image, m=mrfterm, priorweight=0.25, c=atroposits,
                i=seginit['probabilityimages'], x=mask)

    # 5 curvature seg - use results of 4 to segment based on
    # 5.1 gm/wm  image
    # 5.2 gm/csf image
    # done!

    return {
        'segobj': s3,
        'seginit': seginit,
        'thksig': thksig,
        'thkj': thkj,
        'jacw': jacw
    }


