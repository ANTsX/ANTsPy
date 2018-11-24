"""
Atropos segmentation
"""



__all__ = ['atropos']

import os
import glob
import warnings
from tempfile import mktemp

from ..core import ants_image_io as iio2
from .. import utils


def atropos(a, x, i='Kmeans[3]', m='[0.2,1x1]', c='[5,0]',
            priorweight=0.25, **kwargs):
    """
    A finite mixture modeling (FMM) segmentation approach with possibilities
    for specifying prior constraints. These prior constraints include the
    specification of a prior label image, prior probability images (one for
    each class), and/or an MRF prior to enforce spatial smoothing of the
    labels. Similar algorithms include FAST and SPM. atropos can also perform
    multivariate segmentation if you pass a list of images in: e.g. a=(img1,img2).

    ANTsR function: `atropos`

    Arguments
    ---------
    a : ANTsImage or list/tuple of ANTsImage types
        One or more scalar images to segment. If priors are not used,
        the intensities of the first image are used to order the classes
        in the segmentation output, from lowest to highest intensity. Otherwise
        the order of the classes is dictated by the order of the prior images.

    x : ANTsImage
        mask image.

    i : string
        initialization usually KMeans[N] for N classes or a list of N prior
        probability images. See Atropos in ANTs for full set of options.

    m : string
        mrf parameters as a string, usually "[smoothingFactor,radius]" where
        smoothingFactor determines the amount of smoothing and radius determines
        the MRF neighborhood, as an ANTs style neighborhood vector eg "1x1x1"
        for a 3D image. The radius must match the dimensionality of the image,
        eg 1x1 for 2D and The default in ANTs is smoothingFactor=0.3 and
        radius=1. See Atropos for more options.

    c : string
        convergence parameters, "[numberOfIterations,convergenceThreshold]".
        A threshold of 0 runs the full numberOfIterations, otherwise Atropos
        tests convergence by comparing the mean maximum posterior probability
        over the whole region of interest defined by the mask x.

    priorweight : scalar
        usually 0 (priors used for initialization only), 0.25 or 0.5.

    kwargs : keyword arguments
        more parameters, see Atropos help in ANTs

    Returns
    -------
    dictionary with the following key/value pairs:
        `segmentation`: ANTsImage
            actually segmented image
        `probabilityimages` : list of ANTsImage types
            one image for each segmentation class

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> img = ants.resample_image(img, (64,64), 1, 0)
    >>> mask = ants.get_mask(img)
    >>> ants.atropos( a = img, m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )
    >>> seg2 = ants.atropos( a = img, m = '[0.2,1x1]', c = '[2,0]', i = seg['probabilityimages'], x = mask, priorweight=0.25 )
    """
    probs = mktemp(prefix='antsr', suffix='prob%02d.nii.gz')
    tdir = probs.replace(os.path.basename(probs),'')
    probsbase = os.path.basename(probs)
    searchpattern = probsbase.replace('%02d', '*')

    ct = 0
    if isinstance(i, (list,tuple)) and (len(i) > 1):
        while ct < len(i):
            probchar = str(ct+1)
            if ct < 10:
                probchar = '0%s' % probchar
            tempfn = probs.replace('%02d', probchar)
            iio2.image_write(i[ct], tempfn)
            ct += 1
        i = 'PriorProbabilityImages[%s,%s,%s]' % (str(len(i)), probs, str(priorweight))

    if isinstance(a, (list,tuple)):
        outimg = a[0].clone('unsigned int')
    else:
        outimg = a.clone('unsigned int')

    mydim = outimg.dimension
    outs = '[%s,%s]' % (utils._ptrstr(outimg.pointer), probs)
    mymask = x.clone('unsigned int')

    if (not isinstance(a, (list,tuple))) or (len(a) == 1):
        myargs = {
            'd': mydim,
            'a': a,
            'm-MULTINAME-0': m,
            'o': outs,
            'c': c,
            'm-MULTINAME-1': m,
            'i': i,
            'x': mymask
        }
        for k, v in kwargs.items():
            myargs[k] = v

    elif isinstance(a, (list, tuple)):
        if len(a) > 6:
            print('more than 6 input images not really supported, using first 6')
            a = a[:6]
        myargs = {
            'd': mydim,
            'm-MULTINAME-0': m,
            'o': outs,
            'c': c,
            'm-MULTINAME-1': m,
            'i': i,
            'x': mymask
        }
        for aa_idx, aa in enumerate(a):
            myargs['a-MULTINAME-%i'%aa_idx] = aa

    processed_args = utils._int_antsProcessArguments(myargs)
    libfn = utils.get_lib_fn('Atropos')
    retval = libfn(processed_args)

    if retval != 0:
        warnings.warn('ERROR: Non-zero exit status!')

    probsout = glob.glob(os.path.join(tdir,'*'+searchpattern))
    probsout.sort()
    probimgs = [iio2.image_read(probsout[0])]
    for idx in range(1, len(probsout)):
        probimgs.append(iio2.image_read(probsout[idx]))

    outimg = outimg.clone('float')
    return {'segmentation': outimg,
            'probabilityimages': probimgs}
