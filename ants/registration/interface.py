"""
ANTsPy Registration
"""
__all__ = ['registration']

import os
import numpy as np
from tempfile import mktemp
import glob
import re

from .. import utils
from ..core import ants_image as iio


def registration(fixed,
                 moving,
                 type_of_transform='SyN',
                 initial_transform=None,
                 outprefix='',
                 mask=None,
                 grad_step=0.2,
                 flow_sigma=3,
                 total_sigma=0,
                 aff_metric='mattes',
                 aff_sampling=32,
                 syn_metric='mattes',
                 syn_sampling=32,
                 reg_iterations=(40,20,0),
                 verbose=False,
                 **kwargs):
    """
    Register a pair of images either through the full or simplified 
    interface to the ANTs registration method.
    
    ANTsR function: `antsRegistration`

    Arguments
    ---------
    fixed : ANTsImage
        fixed image to which we register the moving image.
    
    moving : ANTsImage
        moving image to be mapped to fixed space.
    
    type_of_transform : string
        A linear or non-linear registration type. Mutual information metric by default. 
        See Notes below for more.
    
    initial_transform : list of strings (optional)  
        transforms to prepend
    
    outprefix : string
        output will be named with this prefix.
    
    mask : ANTsImage (optional)
        mask the registration.
    
    grad_step : scalar
        gradient step size (not for all tx)
    
    flow_sigma : scalar  
        smoothing for update field
    
    total_sigma : scalar 
        smoothing for total field
    
    aff_metric : string  
        the metric for the affine part (GC, mattes, meansquares)
    
    aff_sampling : scalar
        the nbins or radius parameter for the syn metric
    
    syn_metric : string  
        the metric for the syn part (CC, mattes, meansquares, demons)
    
    syn_sampling : scalar
        the nbins or radius parameter for the syn metric
    
    reg_iterations : list/tuple of integers  
        vector of iterations for syn. we will set the smoothing and multi-resolution parameters based on the length of this vector.
    
    verbose : boolean
        request verbose output (useful for debugging)
    
    kwargs : keyword args
        extra arguments 
    
    Returns
    -------
    dict containing follow key/value pairs:
        `warpedmovout`: Moving image warped to space of fixed image.
        `warpedfixout`: Fixed image warped to space of moving image.
        `fwdtransforms`: Transforms to move from moving to fixed image.
        `invtransforms`: Transforms to move from fixed to moving image.

    Notes
    -----
    typeofTransform can be one of:
        - "Translation": Translation transformation.
        - "Rigid": Rigid transformation: Only rotation and translation.
        - "Similarity": Similarity transformation: scaling, rotation and translation.
        - "QuickRigid": Rigid transformation: Only rotation and translation. 
                        May be useful for quick visualization fixes.'
        - "DenseRigid": Rigid transformation: Only rotation and translation. 
                        Employs dense sampling during metric estimation.'
        - "BOLDRigid": Rigid transformation: Parameters typical for BOLD to 
                        BOLD intrasubject registration'.'
        - "Affine": Affine transformation: Rigid + scaling.
        - "AffineFast": Fast version of Affine.
        - "BOLDAffine": Affine transformation: Parameters typical for BOLD to 
                        BOLD intrasubject registration'.'
        - "TRSAA": translation, rigid, similarity, affine (twice). please set 
                    regIterations if using this option. this would be used in 
                    cases where you want a really high quality affine mapping 
                    (perhaps with mask).
        - "ElasticSyN": Symmetric normalization: Affine + deformable 
                        transformation, with mutual information as optimization 
                        metric and elastic regularization.
        - "SyN": Symmetric normalization: Affine + deformable transformation, 
                    with mutual information as optimization metric.
        - "SyNRA": Symmetric normalization: Rigid + Affine + deformable 
                    transformation, with mutual information as optimization metric.
        - "SyNOnly": Symmetric normalization: no initial transformation, 
                    with mutual information as optimization metric. Assumes 
                    images are aligned by an inital transformation. Can be 
                    useful if you want to run an unmasked affine followed by 
                    masked deformable registration.
        - "SyNCC": SyN, but with cross-correlation as the metric.
        - "SyNabp": SyN optimized for abpBrainExtraction.
        - "SyNBold": SyN, but optimized for registrations between BOLD and T1 images.
        - "SyNBoldAff": SyN, but optimized for registrations between BOLD 
                        and T1 images, with additional affine step.
        - "SyNAggro": SyN, but with more aggressive registration 
                        (fine-scale matching and more deformation). 
                        Takes more time than SyN.
        - "TVMSQ": time-varying diffeomorphism with mean square metric
        - "TVMSQC": time-varying diffeomorphism with mean square metric for very large deformation

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('r16'))
    >>> mi = ants.image_read(ants.get_ants_data('r64'))
    >>> fi = ants.resample_image(fi, (60,60), 1, 0)
    >>> mi = ants.resample_image(mi, (60,60), 1, 0)
    >>> mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'SyN' )
    """
    if isinstance(fixed, list) and (moving is None):
        processed_args = utils._int_antsProcessArguments(fixed)
        libfn = utils.get_lib_fn('antsRegistration')
        libfn(processed_args)
        return 0

    if type_of_transform == '':
        type_of_transform = 'SyN'

    if isinstance(type_of_transform, (tuple,list)) and (len(type_of_transform) == 1):
        type_of_transform = type_of_transform[0]

    if (outprefix == '') or len(outprefix) == 0:
        outprefix = mktemp()

    if (np.sum(np.isnan(fixed.numpy())) > 0):
        raise ValueError('fixed image has NaNs - replace these')
    if (np.sum(np.isnan(moving.numpy())) > 0):
        raise ValueError('moving image has NaNs - replace these')
    #----------------------------

    args = [fixed, moving, type_of_transform, outprefix]
    myl = 0
    myf_aff = '6x4x2x1'
    mys_aff = '3x2x1x0'
    metsam = 0.2

    myiterations = '2100x1200x1200x10'
    if (type_of_transform == 'AffineFast'):
        type_of_transform = 'Affine'
        myiterations = '2100x1200x0x0'
    if (type_of_transform == 'BOLDAffine'):
        type_of_transform = 'Affine'
        myf_aff='2x1'
        mys_aff='1x0'
        myiterations = '100x20'
        myl=1
    if (type_of_transform == 'QuickRigid'):
        type_of_transform = 'Rigid'
        myiterations = '20x20x0x0'
    if (type_of_transform == 'DenseRigid'):
        type_of_transform = 'Rigid'
        metsam = 0.8
    if (type_of_transform == 'BOLDRigid'):
        type_of_transform = 'Rigid'
        myf_aff='2x1'
        mys_aff='1x0'
        myiterations = '100x20'
        myl=1

    mysyn = 'SyN[%f,%f,%f]' % (grad_step, flow_sigma, total_sigma)
    itlen = len(reg_iterations)# NEED TO CHECK THIS
    if itlen == 0:
        smoothingsigmas = 0
        shrinkfactors = 1
        synits = reg_iterations
    else:
        smoothingsigmas = np.arange(0, itlen)[::-1].astype('float32') # NEED TO CHECK THIS
        shrinkfactors = 2**smoothingsigmas
        shrinkfactors = shrinkfactors.astype('int')
        smoothingsigmas = 'x'.join([str(ss)[0] for ss in smoothingsigmas])
        shrinkfactors = 'x'.join([str(ss) for ss in shrinkfactors])
        synits = 'x'.join([str(ri) for ri in reg_iterations])

    if not isinstance(fixed, str):
        if isinstance(fixed, iio.ANTsImage) and isinstance(moving, iio.ANTsImage):
            inpixeltype = fixed.pixeltype
            ttexists = False
            allowable_tx =  {'SyNBold','SyNBoldAff', 'ElasticSyn','SyN','SyNRA',
                            'SyNOnly','SyNAggro','SyNCC','TRSAA','SyNabp','SyNLessAggro',
                            'TVMSQ','TVMSQC','Rigid','Similarity','Translation','Affine',
                            'AffineFast','BOLDAffine','QuickRigid','DenseRigid','BOLDRigid'}
            ttexists = type_of_transform in allowable_tx
            if not ttexists:
                raise ValueError('`type_of_transform` does not exist')

            if ttexists:
                initx = initial_transform
                #if isinstance(initx, ANTsTransform):
                    #tempTXfilename = tempfile( fileext = '.mat' )
                    #initx = invertAntsrTransform( initialTransform )
                    #initx = invertAntsrTransform( initx )
                    #writeAntsrTransform( initx, tempTXfilename )
                    #initx = tempTXfilename
                moving = moving.clone('float')
                fixed = fixed.clone('float')
                warpedfixout = moving.clone()
                warpedmovout = fixed.clone()
                f = utils.get_pointer_string(fixed)
                m = utils.get_pointer_string(moving)
                wfo = utils.get_pointer_string(warpedfixout)
                wmo = utils.get_pointer_string(warpedmovout)
                if mask is not None:
                    mask_scale = mask - mask.min()
                    mask_scale = mask_scale / mask_scale.max() * 255.
                    charmask = mask_scale.clone('unsigned char')
                    maskopt = '[%s,NA]' % (utils.get_pointer_string(charmask))
                else:
                    maskopt = None
                if initx is None:
                    initx = '[%s,%s,1]' % (f, m)
                # ------------------------------------------------------------
                if type_of_transform == 'SyNBold':
                    args = ['-d', str(fixed.dimension),
                        '-r', initx,
                        '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                        '-t', 'Rigid[0.25]',
                        '-c', '[1200x1200x100,1e-6,5]',
                        '-s', '2x1x0',
                        '-f', '4x2x1',
                        '-x', '[NA,NA]',
                        '-m', '%s[%s,%s,1,%s]' % (syn_metric, f, m, syn_sampling),
                        '-t', mysyn,
                        '-c', '[%s,1e-7,8]' % synits,
                        '-s', smoothingsigmas,
                        '-f', shrinkfactors,
                        '-u', '1',
                        '-z', '1',
                        '-l', myl,
                        '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'SyNBoldAff':
                    args = ['-d', str(fixed.dimension),
                        '-r', initx,
                        '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                        '-t', 'Rigid[0.25]',
                        '-c', '[1200x1200x100,1e-6,5]',
                        '-s', '2x1x0',
                        '-f', '4x2x1',
                        '-x', '[NA,NA]',
                        '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                        '-t', 'Affine[0.25]',
                        '-c', '[200x20,1e-6,5]',
                        '-s', '1x0',
                        '-f', '2x1',
                        '-x', '[NA,NA]',
                        '-m', '%s[%s,%s,1,%s]' % (syn_metric, f, m, syn_sampling),
                        '-t', mysyn,
                        '-c', '[%s,1e-7,8]' % (synits),
                        '-s', smoothingsigmas,
                        '-f', shrinkfactors,
                        '-u', '1',
                        '-z', '1',
                        '-l', myl,
                        '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'ElasticSyn':
                    args = ['-d', str(fixed.dimension), 
                        '-r', initx,
                        '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                        '-t', 'Affine[0.25]', 
                        '-c', '2100x1200x200x0', 
                        '-s', '3x2x1x0',
                        '-f', '4x2x2x1',
                        '-x', '[NA,NA]',
                        '-m', '%s[%s,%s,1,%s]' % (syn_metric, f, m, syn_sampling),
                        '-t', mysyn,
                        '-c', '[%s,1e-7,8]' % (synits),
                        '-s', smoothingsigmas, 
                        '-f', shrinkfactors,
                        '-u', '1', 
                        '-z', '1', 
                        '-l', myl,
                        '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'SyN':
                    args = ['-d', str(fixed.dimension),
                        '-r', initx,
                        '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                        '-t', 'Affine[0.25]',
                        '-c', '2100x1200x1200x0',
                        '-s', '3x2x1x0',
                        '-f', '4x2x2x1',
                        '-x', '[NA,NA]',
                        '-m', '%s[%s,%s,1,%s]' % (syn_metric, f, m, syn_sampling),
                        '-t', '%s[0.25,3,0]' % type_of_transform,
                        '-c', '[%s,1e-7,8]' % synits,
                        '-s', smoothingsigmas,
                        '-f', shrinkfactors,
                        '-u', '1',
                        '-z', '1',
                        '-l', myl,
                        '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'SyNRA':
                    args = ['-d', str(fixed.dimension), 
                            '-r', initx,
                            '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Rigid[0.25]', 
                            '-c', '2100x1200x1200x0', 
                            '-s', '3x2x1x0',
                            '-f', '4x2x2x1',
                            '-x', '[NA,NA]',
                            '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Affine[0.25]', 
                            '-c', '2100x1200x1200x0', 
                            '-s', '3x2x1x0',
                            '-f', '4x2x2x1',
                            '-x', '[NA,NA]',
                            '-m', '%s[%s,%s,1,%s]' % (syn_metric, f, m, syn_sampling),
                            '-t', mysyn,
                            '-c', '[%s,1e-7,8]' % synits,
                            '-s', smoothingsigmas, 
                            '-f', shrinkfactors,
                            '-u', '1', 
                            '-z', '1', 
                            '-l', myl, 
                            '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'SyNOnly':
                    args = ['-d', str(fixed.dimension), 
                            '-r', initx,
                            '-m', '%s[%s,%s,1,%s]' % (syn_metric, f, m, syn_sampling),
                            '-t', mysyn,
                            '-c', '[%s,1e-7,8]' % synits,
                            '-s', smoothingsigmas, 
                            '-f', shrinkfactors,
                            '-u', '1', 
                            '-z', '1', 
                            '-l', myl, 
                            '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'SyNAggro':
                    args = ['-d', str(fixed.dimension), 
                            '-r', initx,
                            '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Affine[0.25]', 
                            '-c', '2100x1200x1200x100', 
                            '-s', '3x2x1x0',
                            '-f', '4x2x2x1',
                            '-x', '[NA,NA]',
                            '-m', '%s[%s,%s,1,%s]' % (syn_metric, f, m, syn_sampling),
                            '-t', mysyn,
                            '-c', '[%s,1e-7,8]' % synits,
                            '-s', smoothingsigmas,
                            '-f', shrinkfactors, 
                            '-u', '1', 
                            '-z', '1', 
                            '-l', myl,
                            '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'SyNCC':
                    syn_metric = 'CC'
                    syn_sampling = 4
                    synits = '2100x1200x1200x20'
                    smoothingsigmas = '3x2x1x0'
                    shrinkfactors = '4x3x2x1'
                    mysyn = 'SyN[0.15,3,0]'

                    args = ['-d', str(fixed.dimension), 
                            '-r', initx,
                            '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Rigid[1]',
                            '-c', '2100x1200x1200x0',
                            '-s', '3x2x1x0',
                            '-f', '4x4x2x1',
                            '-x', '[NA,NA]',
                            '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Affine[1]', 
                            '-c', '1200x1200x100', 
                            '-s', '2x1x0',
                            '-f', '4x2x1',
                            '-x', '[NA,NA]',
                            '-m', '%s[%s,%s,1,%s]' % (syn_metric, f, m, syn_sampling),
                            '-t', mysyn,
                            '-c', '[%s,1e-7,8]' % synits,
                            '-s', smoothingsigmas,
                            '-f', shrinkfactors, 
                            '-u', '1', 
                            '-z', '1', 
                            '-l', myl,
                            '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'TRSAA':
                    itlen = len( reg_iterations )
                    itlenlow = round( itlen/2 + 0.0001 )
                    dlen = itlen - itlenlow
                    _myconvlow = [2000]*itlenlow + [0]*dlen
                    myconvlow = 'x'.join([str(mc) for mc in _myconvlow])
                    myconvhi = 'x'.join([str(r) for r in reg_iterations])
                    myconvhi = '[%s,1.e-7,10]' % myconvhi
                    args = ['-d', str(fixed.dimension), 
                            '-r', initx,
                            '-m', '%s[%s,%s,1,%s,regular,0.3]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Translation[1]',
                            '-c', myconvlow,
                            '-s', smoothingsigmas,
                            '-f', shrinkfactors,
                            '-x', '[NA,NA]',
                            '-m', '%s[%s,%s,1,%s,regular,0.3]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Rigid[1]',
                            '-c', myconvlow,
                            '-s', smoothingsigmas,
                            '-f', shrinkfactors,
                            '-x', '[NA,NA]',
                            '-m', '%s[%s,%s,1,%s,regular,0.3]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Similarity[1]',
                            '-c', myconvlow,
                            '-s', smoothingsigmas,
                            '-f', shrinkfactors,
                            '-x', '[NA,NA]',
                            '-m', '%s[%s,%s,1,%s,regular,0.3]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Affine[1]',
                            '-c', myconvhi,
                            '-s', smoothingsigmas,
                            '-f', shrinkfactors,
                            '-x', '[NA,NA]',
                            '-m', '%s[%s,%s,1,%s,regular,0.3]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Affine[1]',
                            '-c', myconvhi,
                            '-s', smoothingsigmas,
                            '-f', shrinkfactors,
                            '-u', '1', 
                            '-z', '1', 
                            '-l', myl,
                            '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------s
                elif type_of_transform == 'SyNabp':
                    args = ['-d', str(fixed.dimension), 
                            '-r', initx,
                            '-m', 'mattes[%s,%s,1,32,regular,0.25]' % (f, m),
                            '-t', 'Rigid[0.1]', 
                            '-c', '1000x500x250x100', 
                            '-s', '4x2x1x0',
                            '-f', '8x4x2x1',
                            '-x', '[NA,NA]',
                            '-m', 'mattes[%s,%s,1,32,regular,0.25]' % (f, m),
                            '-t', 'Affine[0.1]', 
                            '-c', '1000x500x250x100', 
                            '-s', '4x2x1x0',
                            '-f', '8x4x2x1',
                            '-x', '[NA,NA]',
                            '-m', 'CC[%s,%s,0.5,4]' % (f,m),
                            '-t', 'SyN[0.1,3,0]',
                            '-c', '50x10x0',
                            '-s', '2x1x0', 
                            '-f', '4x2x1', 
                            '-u', '1', 
                            '-z', '1', 
                            '-l', myl,
                            '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'SyNLessAggro':
                    args = ['-d', str(fixed.dimension),
                            '-r', initx,
                            '-m', '%s[%s,%s,1,%s,regular,0.2]' % (aff_metric, f, m, aff_sampling),
                            '-t', 'Affine[0.25]', 
                            '-c', '2100x1200x1200x100', 
                            '-s', '3x2x1x0',
                            '-f', '4x2x2x1',
                            '-x', '[NA,NA]',
                            '-m', '%s[%s,%s,1,%s]' % (syn_metric, f, m, syn_sampling),
                            '-t', mysyn,
                            '-c', '[%s,1e-7,8]' % synits,
                            '-s', smoothingsigmas,
                            '-f', shrinkfactors, 
                            '-u', '1', 
                            '-z', '1', 
                            '-l', myl,
                            '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'TVMSQ':
                    if grad_step is None:
                        grad_step = 1.0

                    tvtx = 'TimeVaryingVelocityField[%s, 4, 0.0,0.0, 0.5,0 ]' % str(grad_step)
                    args = ['-d', str(fixed.dimension),
                            # '-r', initx,
                            '-m', '%s[%s,%s,1,%s]' % (syn_metric, f, m, syn_sampling),
                            '-t', tvtx,
                            '-c', '[%s,1e-7,8]' % synits,
                            '-s', smoothingsigmas,
                            '-f', shrinkfactors,
                            '-u', '1', 
                            '-z', '1', 
                            '-l', myl,
                            '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif type_of_transform == 'TVMSQC':
                    if grad_step is None:
                        grad_step = 2.0

                    tvtx = 'TimeVaryingVelocityField[%s, 8, 1.0,0.0, 0.05,0 ]' % str(grad_step)
                    args = ['-d', str(fixed.dimension),
                            # '-r', initx,
                            '-m', 'demons[%s,%s,0.5,0]' % (f, m),
                            '-m', 'meansquares[%s,%s,1,0]' % (f, m),
                            '-t', tvtx,
                            '-c', '[1200x1200x100x20x0,0,5]',
                            '-s', '8x6x4x2x1vox',
                            '-f', '8x6x4x2x1',
                            '-u', '1', 
                            '-z', '1', 
                            '-l', myl,
                            '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                elif (type_of_transform == 'Rigid') or (type_of_transform == 'Similarity') or \
                     (type_of_transform == 'Translation') or (type_of_transform == 'Affine'):
                    args = ['-d', str(fixed.dimension),
                            '-r', initx,
                            '-m', '%s[%s,%s,1,%s,regular,%s]' % (aff_metric, f, m, aff_sampling, metsam),
                            '-t', '%s[0.25]' % type_of_transform,
                            '-c', myiterations,
                            '-s', mys_aff, 
                            '-f', myf_aff, 
                            '-u', '1', 
                            '-z', '1', 
                            '-l', myl,
                            '-o', '[%s,%s,%s]' % (outprefix, wmo, wfo)]
                    if maskopt is not None:
                        args.append('-x')
                        args.append(maskopt)
                    else:
                        args.append('-x')
                        args.append('[NA,NA]')
                # ------------------------------------------------------------
                args.append('--float')
                args.append('1')
                if verbose:
                    args.append('-v')
                    args.append('1')

                processed_args = utils._int_antsProcessArguments(args)
                libfn = utils.get_lib_fn('antsRegistration')
                libfn(processed_args)
                afffns = glob.glob(outprefix+'*'+'[0-9]GenericAffine.mat')
                fwarpfns = glob.glob(outprefix+'*'+'[0-9]Warp.nii.gz')
                iwarpfns = glob.glob(outprefix+'*'+'[0-9]InverseWarp.nii.gz')
                #print(afffns, fwarpfns, iwarpfns)
                if len(afffns) == 0:
                    afffns = ''
                if len(fwarpfns) == 0:
                    fwarpfns = ''
                if len(iwarpfns) == 0:
                    iwarpfns = ''

                alltx = sorted(glob.glob(outprefix+'*'+'[0-9]*'))
                findinv = np.where([re.search('[0-9]InverseWarp.nii.gz',ff) for ff in alltx])[0]
                findfwd = np.where([re.search('[0-9]Warp.nii.gz', ff) for ff in alltx])[0]
                if len(findinv) > 0:
                    fwdtransforms = list(reversed([ff for idx,ff in enumerate(alltx) if idx !=findinv[0]]))
                    invtransforms = [ff for idx,ff in enumerate(alltx) if idx!=findfwd[0]]
                else:
                    fwdtransforms = list(reversed(alltx))
                    invtransforms = alltx

                return {
                    'warpedmovout': warpedmovout.clone(inpixeltype),
                    'warpedfixout': warpedfixout.clone(inpixeltype),
                    'fwdtransforms': fwdtransforms,
                    'invtransforms': invtransforms
                }
    else:
        args.append('--float')
        args.append('1')
        if verbose:
            args.append('-v')
            args.append('1')
            processed_args = utils._int_antsProcessArguments(args)
            libfn = utils.get_lib_fn('antsRegistration')
            libfn(processed_args)
            return 0
    
