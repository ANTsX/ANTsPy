"""
Joint Label Fusion algorithm
"""

__all__ = ['joint_label_fusion']

import os
import numpy as np

from tempfile import mktemp
import glob
import re

from .. import utils
from ..core import ants_image as iio
from ..core import ants_image_io as iio2


def joint_label_fusion(target_image, target_image_mask, atlas_list, beta=4, rad=2,
                        label_list=None, rho=0.01, usecor=False, r_search=3, 
                        nonnegative=False, verbose=False):
    """
    A multiple atlas voting scheme to customize labels for a new subject. 
    This function will also perform intensity fusion. It almost directly 
    calls the C++ in the ANTs executable so is much faster than other 
    variants in ANTsR. 

    One may want to normalize image intensities for each input image before 
    passing to this function. If no labels are passed, we do intensity fusion. 
    Note on computation time: the underlying C++ is multithreaded. 
    You can control the number of threads by setting the environment 
    variable ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS e.g. to use all or some 
    of your CPUs. This will improve performance substantially. 
    For instance, on a macbook pro from 2015, 8 cores improves speed by about 4x.
    
    ANTsR function: `jointLabelFusion`

    Arguments
    ---------
    target_image : ANTsImage
        image to be approximated
    
    target_image_mask : ANTsImage
        mask with value 1
    
    atlas_list : list of ANTsImage types
        list containing intensity images
    
    beta : scalar  
        weight sharpness, default to 2
    
    rad : scalar
        neighborhood radius, default to 2
    
    label_list : list of ANTsImage types (optional)   
        list containing images with segmentation labels
    
    rho : scalar
        ridge penalty increases robustness to outliers but also makes image converge to average
    
    usecor : boolean
        employ correlation as local similarity
    
    r_search : scalar
        radius of search, default is 3
    
    nonnegative : boolean
        constrain weights to be non-negative
    
    verbose : boolean
        whether to show status updates

    Returns
    -------
    dictionary w/ following key/value pairs:
        `segmentation` : ANTsImage
            segmentation image
        
        `intensity` : ANTsImage
            intensity image
        
        `probabilityimages` : list of ANTsImage types
            probability map image for each label

    Example
    -------
    >>> import ants
    >>> ref = ants.image_read( ants.get_ants_data('r16'))
    >>> ref = ants.resample_image(ref, (50,50),1,0)
    >>> ref = ants.iMath(ref,'Normalize')
    >>> mi = ants.image_read( ants.get_ants_data('r27'))
    >>> mi2 = ants.image_read( ants.get_ants_data('r30'))
    >>> mi3 = ants.image_read( ants.get_ants_data('r62'))
    >>> mi4 = ants.image_read( ants.get_ants_data('r64'))
    >>> mi5 = ants.image_read( ants.get_ants_data('r85'))
    >>> refmask = ants.get_mask(ref)
    >>> refmask = ants.iMath(refmask,'ME',2) # just to speed things up
    >>> ilist = [mi,mi2,mi3,mi4,mi5]
    >>> seglist = [None]*len(ilist)
    >>> for i in range(len(ilist)):
    >>>     ilist[i] = ants.iMath(ilist[i],'Normalize')
    >>>     mytx = ants.registration(fixed=ref , moving=ilist[i] ,
    >>>         typeofTransform = ('Affine') )
    >>>     mywarpedimage = ants.apply_transforms(fixed=ref,moving=ilist[i],
    >>>             transformlist=mytx['fwdtransforms'])
    >>>     ilist[i] = mywarpedimage
    >>>     seg = ants.threshold_image(ilist[i],'Otsu', 3)
    >>>     seglist[i] = seg
    >>> r = 2
    >>> pp = ants.joint_label_fusion(ref, refmask, ilist, r_search=2,
    >>>                     label_list=seglist, rad=[r]*ref.dimension )
    >>> pp = ants.joint_label_fusion(ref,refmask,ilist, r_search=2, rad=[r]*ref.dimension)
    """
    segpixtype = 'unsigned int'
    if np.any([l is None for l in label_list]):
        doJif = True
    else:
        doJif = False

    if not doJif:
        if len(label_list) != len(atlas_list):
            raise ValueError('len(label_list) != len(atlas_list)')
        inlabs = np.sort(np.unique(label_list[0][target_image_mask == 1]))
        labsum = label_list[0]
        for n in range(1, len(label_list)):
            inlabs = np.sort(np.unique(np.hstack([inlabs, label_list[n][target_image_mask==1]])))
            labsum = labsum + label_list[n]

        mymask = target_image_mask.clone()
        mymask[labsum==0] = 0
    else:
        mymask = [target_image_mask]

    osegfn = mktemp(prefix='antsr', suffix='myseg.nii.gz')
    #segdir = osegfn.replace(os.path.basename(osegfn),'')

    if os.path.exists(osegfn):
        os.remove(osegfn)

    probs = mktemp(prefix='antsr', suffix='prob%02d.nii.gz')
    probsbase = os.path.basename(probs)
    tdir = probs.replace(probsbase,'')
    searchpattern = probsbase.replace('%02d', '*')

    mydim = target_image_mask.dimension
    if not doJif:
        # not sure if these should be allocated or what their size should be
        outimg = iio2.make_image(imagesize=[128]*mydim, pixeltype=segpixtype)
        outimgi = iio2.make_image(imagesize=[128]*mydim, pixeltype='float')

        outimg_ptr = utils.get_pointer_string(outimg)
        outimgi_ptr = utils.get_pointer_string(outimgi)
        outs = '[%s,%s,%s]' % (outimg_ptr, outimgi_ptr, probs)
    else:
        outimgi = iio2.make_image(imagesize=[128]*mydim, pixeltype='float')
        outs = utils.get_pointer_string(outimgi)

    mymask = mymask.clone(segpixtype)
    if (not isinstance(rad, (tuple,list))) or (len(rad)==1):
        myrad = [rad]*mydim
    else:
        myrad = rad

    if len(myrad) != mydim:
        raise ValueError('path radius dimensionality must equal image dimensionality')

    myrad = 'x'.join([str(mr) for mr in myrad])
    vnum = 1 if verbose else 0
    nnum = 1 if nonnegative else 0

    myargs = {
        'd': mydim,
        't': target_image,
        'a': rho,
        'b': beta,
        'c': nnum,
        'p': myrad,
        'm': 'PC',
        's': r_search,
        'x': mymask,
        'o': outs,
        'v': vnum
    }

    kct = len(myargs.keys())
    for k in range(len(atlas_list)):
        kct += 1
        myargs['g-MULTINAME-%i' % kct] = atlas_list[k]
        if not doJif:
            kct += 1
            castseg = label_list[k].clone(segpixtype)
            myargs['l-MULTINAME-%i' % kct] = castseg

    myprocessedargs = utils._int_antsProcessArguments(myargs)
    
    libfn = utils.get_lib_fn('antsJointFusion')
    rval = libfn(myprocessedargs)
    if rval != 0:
        print('Warning: Non-zero return from antsJointFusion')

    if doJif:
        return outimgi

    probsout = glob.glob(os.path.join(tdir,'*'+searchpattern))
    probimgs = [iio2.image_read(probsout[0])]
    for idx in range(1, len(probsout)):
        probimgs.append(iio2.image_read(probsout[idx]))

    segmat = iio2.images_to_matrix(probimgs, target_image_mask)
    finalsegvec = segmat.argmax(axis=0)
    finalsegvec2 = finalsegvec

    # mapfinalsegvec to original labels
    for i in range(finalsegvec.max()):
        finalsegvec2[finalsegvec==i] = inlabs[i]

    outimg = iio2.make_image(target_image_mask, finalsegvec2)

    return {
        'segmentation': outimg,
        'intensity': outimgi,
        'probabilityimages': probimgs
    }





