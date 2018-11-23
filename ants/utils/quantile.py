
__all__ = ['quantile',
           'regress_poly',
           'regress_components',
           'get_average_of_timeseries',
           'compcor' ]

import numpy as np
from numpy.polynomial import Legendre
from scipy import linalg
from .. import utils
from .. import core

def quantile(image, q, nonzero=True):
    """
    Get the quantile values from an ANTsImage
    """
    img_arr = image.numpy()
    if isinstance(q, (list,tuple)):
        q = [qq*100. if qq <= 1. else qq for qq in q]
        if nonzero:
            img_arr = img_arr[img_arr>0]
        vals = [np.percentile(img_arr, qq) for qq in q]
        return tuple(vals)
    elif isinstance(q, (float,int)):
        if q <= 1.:
            q = q*100.
        if nonzero:
            img_arr = img_arr[img_arr>0]
        return np.percentile(img_arr[img_arr>0], q)
    else:
        raise ValueError('q argument must be list/tuple or float/int')


def regress_poly(degree, data, remove_mean=True, axis=-1):
    """
    Returns data with degree polynomial regressed out.
    :param bool remove_mean: whether or not demean data (i.e. degree 0),
    :param int axis: numpy array axes along which regression is performed
    """
    timepoints = data.shape[0]
    # Generate design matrix
    X = np.ones((timepoints, 1))  # quick way to calc degree 0
    for i in range(degree):
        polynomial_func = Legendre.basis(i + 1)
        value_array = np.linspace(-1, 1, timepoints)
        X = np.hstack((X, polynomial_func(value_array)[:, np.newaxis]))
    non_constant_regressors = X[:, :-1] if X.shape[1] > 1 else np.array([])
    betas = np.linalg.pinv(X).dot(data)
    if remove_mean:
        datahat = X.dot(betas)
    else:  # disregard the first layer of X, which is degree 0
        datahat = X[:, 1:].dot(betas[1:, ...])
    regressed_data = data - datahat
    return regressed_data, non_constant_regressors

def regress_components( data, components, remove_mean=True ):
    """
    Returns data with components regressed out.
    :param bool remove_mean: whether or not demean data (i.e. degree 0),
    :param int axis: numpy array axes along which regression is performed
    """
    timepoints = data.shape[0]
    betas = np.linalg.pinv(components).dot(data)
    if remove_mean:
        datahat = components.dot(betas)
    else:  # disregard the first layer of X, which is degree 0
        datahat = components[:, 1:].dot(betas[1:, ...])
    regressed_data = data - datahat
    return regressed_data



def get_average_of_timeseries( image, idx=None ):
    """Average the timeseries into a dimension-1 image.
    image: input time series image
    idx:  indices over which to average
    """
    imagedim = image.dimension
    if idx is None:
        idx = range( image.shape[ imagedim - 1 ] )
    i0 = utils.slice_image( image, axis=image.dimension-1, idx=idx[0] ) * 0
    wt = 1.0 / len( idx )
    for k in idx:
        i0 = i0 + utils.slice_image( image, axis=image.dimension-1, idx=k ) * wt
    return( i0 )

def compcor( boldImage, ncompcor=4, quantile=0.975, mask=None, filter_type=False, degree=2 ):
    """Compute the noise components from the imagematrix
    boldImage: input time series image
    ncompcor:  number of noise components to return
    quantile:  quantile defining high-variance
    mask:      mask defining brain or specific tissues
    filter_type: type off filter to apply to time series before computing
                 noise components.
        'polynomial' - Legendre polynomial basis
        False - None (mean-removal only)
    degree: order of polynomial used to remove trends from the timeseries
    returns:
    components: a numpy array
    basis: a numpy array containing the (non-constant) filter regressors

    this is adapted from nipy code https://github.com/nipy/nipype/blob/e29ac95fc0fc00fedbcaa0adaf29d5878408ca7c/nipype/algorithms/confounds.py
    """
    def compute_tSTD(M, quantile, x=0, axis=0):
        stdM = np.std(M, axis=axis)
        # set bad values to x
        stdM[stdM == 0] = x
        stdM[np.isnan(stdM)] = x
        tt = round( quantile*100 )
        threshold_std = np.percentile( stdM, tt )
    #    threshold_std = quantile( stdM, quantile )
        return { 'tSTD': stdM, 'threshold_std': threshold_std}
    if mask is None:
        temp = utils.slice_image( boldImage, axis=boldImage.dimension-1, idx=0 )
        mask = utils.get_mask( temp )
    imagematrix = core.timeseries_to_matrix( boldImage, mask )
    temp = compute_tSTD( imagematrix, quantile, 0 )
    tsnrmask = core.make_image( mask,  temp['tSTD'] )
    tsnrmask = utils.threshold_image( tsnrmask, temp['threshold_std'], temp['tSTD'].max() )
    M = core.timeseries_to_matrix( boldImage, tsnrmask )
    components = None
    basis = np.array([])
    if filter_type in ('polynomial', False):
        M, basis = regress_poly(degree, M)
#    M = M / compute_tSTD(M, 1.)['tSTD']
    # "The covariance matrix C = MMT was constructed and decomposed into its
    # principal components using a singular value decomposition."
    u, _, _ = linalg.svd(M, full_matrices=False)
    if components is None:
        components = u[:, :ncompcor]
    else:
        components = np.hstack((components, u[:, :ncompcor]))
    if components is None and ncompcor > 0:
        raise ValueError('No components found')
    return { 'components': components, 'basis': basis }
