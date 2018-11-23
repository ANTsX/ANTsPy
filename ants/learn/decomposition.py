

__all__ = ['eig_seg',
           'initialize_eigenanatomy',
           'sparse_decom2']

import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from pandas import DataFrame
from sklearn import linear_model
import statsmodels.api as sm

from .. import core
from .. import utils
from ..core import ants_image as iio


def sparse_decom2(inmatrix,
                    inmask=(None, None),
                    sparseness=(0.01, 0.01),
                    nvecs=3,
                    its=20,
                    cthresh=(0,0),
                    statdir=None,
                    perms=0,
                    uselong=0,
                    z=0,
                    smooth=0,
                    robust=0,
                    mycoption=0,
                    initialization_list=[],
                    initialization_list2=[],
                    ell1=10,
                    prior_weight=0,
                    verbose=False,
                    rejector=0,
                    max_based=False,
                    version=1):
    """
    Decomposes two matrices into paired sparse eigenevectors to
    maximize canonical correlation - aka Sparse CCA.
    Note: we do not scale the matrices internally. We leave
    scaling choices to the user.

    ANTsR function: `sparseDecom2`

    Arguments
    ---------
    inmatrix : 2-tuple of ndarrays
        input as inmatrix=(mat1,mat2). n by p input matrix and n by q input matrix , spatial variable lies along columns.

    inmask : 2-tuple of ANTsImage types (optional - one or both)
        optional pair of image masks

    sparseness : tuple
        a pair of float values e.g c(0.01,0.1) enforces an unsigned 99 percent and 90 percent sparse solution for each respective view

    nvecs : integer
        number of eigenvector pairs

    its : integer
        number of iterations, 10 or 20 usually sufficient

    cthresh : 2-tuple
        cluster threshold pair

    statdir : string (optional)
        temporary directory if you want to look at full output

    perms : integer
        number of permutations. settings permutations greater than 0 will estimate significance per vector empirically. For small datasets, these may be conservative. p-values depend on how one scales the input matrices.

    uselong : boolean
        enforce solutions of both views to be the same - requires matrices to be the same size

    z : float
        subject space (low-dimensional space) sparseness value

    smooth : float
        smooth the data (only available when mask is used)

    robust : boolean
        rank transform input matrices

    mycoption : integer
        enforce 1 - spatial orthogonality, 2 - low-dimensional orthogonality or 0 - both

    initialization_list : list
        initialization for first view

    initialization_list2 : list
        initialization for 2nd view

    ell1 : float
        gradient descent parameter, if negative then l0 otherwise use l1

    prior_weight : scalar
        Scalar value weight on prior between 0 (prior is weak) and 1 (prior is strong). Only engaged if initialization is used

    verbose : boolean
        activates verbose output to screen

    rejector : scalar
        rejects small correlation solutions

    max_based : boolean
        whether to choose max-based thresholding

    Returns
    -------
    dict w/ following key/value pairs:

        `projections` : ndarray
            X projections

        `projections2` : ndarray
            Y projections

        `eig1` : ndarray
            X components

        `eig2` : ndarray
            Y components

        `summary` : pd.DataFrame
            first column is canonical correlations,
            second column is p-values (these are `None` if perms > 0)

    Example
    -------
    >>> import numpy as np
    >>> import ants
    >>> mat = np.random.randn(20, 100)
    >>> mat2 = np.random.randn(20, 90)
    >>> mydecom = ants.sparse_decom2(inmatrix = (mat,mat2),
                                    sparseness=(0.1,0.3), nvecs=3,
                                    its=3, perms=0)

    """
    if inmatrix[0].shape[0] != inmatrix[1].shape[0]:
        raise ValueError('Matrices must have same number of rows (samples)')

    idim = 3

    if isinstance(inmask[0], iio.ANTsImage):
        maskx = inmask[0].clone('float')
        idim = inmask[0].dimension
        hasmaskx = 1
    elif isinstance(inmask[0], np.ndarray):
        maskx = core.from_numpy(inmask[0], pixeltype='float')
        idim = inmask[0].ndim
        hasmaskx = 1
    else:
        maskx = core.make_image([1]*idim, pixeltype='float')
        hasmaskx = -1

    if isinstance(inmask[1], iio.ANTsImage):
        masky = inmask[1].clone('float')
        idim = inmask[1].dimension
        hasmasky = 1
    elif isinstance(inmask[1], np.ndarray):
        masky = core.from_numpy(inmask[1], pixeltype='float')
        idim = inmask[1].ndim
        hasmasky = 1
    else:
        masky = core.make_image([1]*idim, pixeltype='float')
        hasmasky = -1

    inmask = [maskx, masky]

    if robust > 0:
        raise NotImplementedError('robust > 0 not currently implemented')
    else:
        input_matrices = inmatrix

    if idim == 2:
        if version == 1:
            sccancpp_fn = utils.get_lib_fn('sccanCpp2D')
        elif version == 2:
            sccancpp_fn = utils.get_lib_fn('sccanCpp2DV2')
            input_matrices = (input_matrices[0].tolist(), input_matrices[1].tolist())
    elif idim ==3:
        if version == 1:
            sccancpp_fn = utils.get_lib_fn('sccanCpp3D')
        elif version == 2:
            sccancpp_fn = utils.get_lib_fn('sccanCpp3DV2')
            input_matrices = (input_matrices[0].tolist(), input_matrices[1].tolist())

    outval = sccancpp_fn(input_matrices[0], input_matrices[1],
                        inmask[0].pointer, inmask[1].pointer,
                        hasmaskx, hasmasky,
                        sparseness[0], sparseness[1],
                        nvecs, its,
                        cthresh[0], cthresh[1],
                        z, smooth,
                        initialization_list, initialization_list2,
                        ell1, verbose,
                        prior_weight, mycoption, max_based)


    p1 = np.dot(input_matrices[0], outval['eig1'].T)
    p2 = np.dot(input_matrices[1], outval['eig2'].T)
    outcorrs = np.array([pearsonr(p1[:,i],p2[:,i])[0] for i in range(p1.shape[1])])
    if prior_weight < 1e-10:
        myord = np.argsort(np.abs(outcorrs))[::-1]
        outcorrs = outcorrs[myord]
        p1 = p1[:, myord]
        p2 = p2[:, myord]
        outval['eig1'] = outval['eig1'][myord,:]
        outval['eig2'] = outval['eig2'][myord,:]

    cca_summary = np.vstack((outcorrs,[None]*len(outcorrs))).T

    if perms > 0:
        cca_summary[:,1] = 0

        nsubs = input_matrices[0].shape[0]
        for permer in range(perms):
            m1 = input_matrices[0][np.random.permutation(nsubs),:]
            m2 = input_matrices[1][np.random.permutation(nsubs),:]
            outvalperm = sccancpp_fn(m1, m2,
                                inmask[0].pointer, inmask[1].pointer,
                                hasmaskx, hasmasky,
                                sparseness[0], sparseness[1],
                                nvecs, its,
                                cthresh[0], cthresh[1],
                                z, smooth,
                                initialization_list, initialization_list2,
                                ell1, verbose,
                                prior_weight, mycoption, max_based)
            p1perm = np.dot(m1, outvalperm['eig1'].T)
            p2perm = np.dot(m2, outvalperm['eig2'].T)
            outcorrsperm = np.array([pearsonr(p1perm[:,i],p2perm[:,i])[0] for i in range(p1perm.shape[1])])
            if prior_weight < 1e-10:
                myord = np.argsort(np.abs(outcorrsperm))[::-1]
                outcorrsperm = outcorrsperm[myord]
            counter = np.abs(cca_summary[:,0]) < np.abs(outcorrsperm)
            counter = counter.astype('int')
            cca_summary[:,1] = cca_summary[:,1] + counter

        cca_summary[:,1] = cca_summary[:,1] / float(perms)

    return {'projections': p1,
            'projections2': p2,
            'eig1': outval['eig1'].T,
            'eig2': outval['eig2'].T,
            'summary': pd.DataFrame(cca_summary,columns=['corrs','pvalues'])}


def initialize_eigenanatomy(initmat, mask=None, initlabels=None, nreps=1, smoothing=0):
    """
    InitializeEigenanatomy is a helper function to initialize sparseDecom
    and sparseDecom2. Can be used to estimate sparseness parameters per
    eigenvector. The user then only chooses nvecs and optional
    regularization parameters.

    Arguments
    ---------
    initmat : np.ndarray or ANTsImage
        input matrix where rows provide initial vector values.
        alternatively, this can be an antsImage which contains labeled regions.

    mask : ANTsImage
        mask if available

    initlabels : list/tuple of integers
        which labels in initmat to use as initial components

    nreps : integer
        nrepetitions to use

    smoothing : float
        if using an initial label image, optionally smooth each roi

    Returns
    -------
    dict w/ the following key/value pairs:
        `initlist` : list of ANTsImage types
            initialization list(s) for sparseDecom(2)

        `mask` : ANTsImage
            mask(s) for sparseDecom(2)

        `enames` : list of strings
            string names of components for sparseDecom(2)

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> mat = np.random.randn(4,100).astype('float32')
    >>> init = ants.initialize_eigenanatomy(mat)
    """
    if isinstance(initmat, iio.ANTsImage):
        # create initmat from each of the unique labels
        if mask is not None:
            selectvec = mask > 0
        else:
            selectvec = initmat > 0
        initmatvec = initmat[selectvec]

        if initlabels is None:
            ulabs = np.sort(np.unique(initmatvec))
            ulabs = ulabs[ulabs > 0]
        else:
            ulabs = initlabels

        nvox = len(initmatvec)
        temp = np.zeros((len(ulabs), nvox))

        for x in range(len(ulabs)):
            timg = utils.threshold_image(initmat, ulabs[x]-1e-4, ulabs[x]+1e-4)
            if smoothing > 0:
                timg = utils.smooth_image(timg, smoothing)
            temp[x,:] = timg[selectvec]
        initmat = temp

    nclasses = initmat.shape[0]
    classlabels = ['init%i'%i for i in range(nclasses)]
    initlist = []
    if mask is None:
        maskmat = np.zeros(initmat.shape)
        maskmat[0,:] = 1
        mask = core.from_numpy(maskmat.astype('float32'))

    eanatnames = ['A'] * (nclasses*nreps)
    ct = 0
    for i in range(nclasses):
        vecimg = mask.clone('float')
        initf = initmat[i,:]
        vecimg[mask==1] = initf
        for nr in range(nreps):
            initlist.append(vecimg)
            eanatnames[ct+nr-1] = str(classlabels[i])
            ct = ct + 1

    return {'initlist': initlist, 'mask': mask, 'enames': eanatnames}



def eig_seg(mask, img_list, apply_segmentation_to_images=False, cthresh=0, smooth=1):
    """
    Segment a mask into regions based on the max value in an image list.
    At a given voxel the segmentation label will contain the index to the image
    that has the largest value. If the 3rd image has the greatest value,
    the segmentation label will be 3 at that voxel.

    Arguments
    ---------
    mask : ANTsImage
        D-dimensional mask > 0 defining segmentation region.

    img_list : collection of ANTsImage or np.ndarray
        images to use

    apply_segmentation_to_images : boolean
        determines if original image list is modified by the segmentation.

    cthresh : integer
        throw away isolated clusters smaller than this value

    smooth : float
        smooth the input data first by this value

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> mylist = [ants.image_read(ants.get_ants_data('r16')),
                  ants.image_read(ants.get_ants_data('r27')),
                  ants.image_read(ants.get_ants_data('r85'))]
    >>> myseg = ants.eig_seg(ants.get_mask(mylist[0]), mylist)
    """
    maskvox = mask > 0
    maskseg = mask.clone()
    maskseg[maskvox] = 0
    if isinstance(img_list, np.ndarray):
        mydata = img_list
    elif isinstance(img_list, (tuple, list)):
        mydata = core.image_list_to_matrix(img_list, mask)

    if (smooth > 0):
        for i in range(mydata.shape[0]):
            temp_img = core.make_image(mask, mydata[i,:], pixeltype='float')
            temp_img = utils.smooth_image(temp_img, smooth, sigma_in_physical_coordinates=True)
            mydata[i,:] = temp_img[mask >= 0.5]

    segids = np.argmax(np.abs(mydata), axis=0)+1
    segmax = np.max(np.abs(mydata), axis=0)
    maskseg[maskvox] = (segids * (segmax > 1e-09))

    if cthresh > 0:
        for kk in range(int(maskseg.max())):
            timg = utils.threshold_image(maskseg, kk, kk)
            timg = utils.label_clusters(timg, cthresh)
            timg = utils.threshold_image(timg, 1, 1e15) * float(kk)
            maskseg[maskseg == kk] = timg[maskseg == kk]

    if (apply_segmentation_to_images) and (not isinstance(img_list, np.ndarray)):
        for i in range(len(img_list)):
            img = img_list[i]
            img[maskseg != float(i)] = 0
            img_list[i] = img

    return maskseg
