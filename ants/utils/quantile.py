
__all__ = ['ilr',
           'rank_intensity',
           'quantile',
           'regress_poly',
           'regress_components',
           'get_average_of_timeseries',
           'compcor',
           'bandpass_filter_matrix' ]

import numpy as np
from numpy.polynomial import Legendre
from scipy import linalg
from scipy.stats import pearsonr
from scipy.stats import rankdata
import pandas as pd
from pandas import DataFrame
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .. import utils
from .. import core


def rank_intensity( x, mask=None, get_mask=True, method='max',  ):
    """
    Rank transform the intensity of the input image with or without masking.
    Intensities will transform from [0,1,2,55] to [0,1,2,3] so this may not be
    appropriate for quantitative images - however, you never know.  rank
    transformations generally improve robustness so it is an empirical question
    that should be evaluated.

    Arguments
    ---------

    x : ANTsImage
        input image

    mask : ANTsImage
        optional mask

    get_mask: boolean
        will estimate a mask when none provided

    method : a scipy rank method (max,min,average,dense)


    return: transformed image

    Example
    -------
    >>> rank_intensity( some_image )
    """
    if mask is not None:
        fir = rankdata( (x*mask).numpy(), method=method )
    elif mask is None and get_mask == True:
        mask = utils.get_mask( x )
        fir = rankdata( (x*mask).numpy(), method=method )
    else:
        fir = rankdata( x.numpy(), method=method )
    fir = fir - 1
    fir = fir.reshape( x.shape )
    rimg = core.from_numpy( fir.astype(float)  )
    rimg = utils.iMath(rimg,"Normalize")
    core.copy_image_info( x, rimg )
    if mask is not None:
        rimg = rimg * mask
    return( rimg )


def ilr( data_frame, voxmats, ilr_formula, verbose = False ):
    """
    Image-based linear regression.

    This function simplifies calculating p-values from linear models
    in which there is a similar formula that is applied many times
    with a change in image-based predictors.  Image-based variables
    are stored in the input matrix list. They should be named
    consistently in the input formula and in the image list.  If they
    are not, an error will be thrown.  All input matrices should have
    the same number of rows and columns.

    This function takes advantage of statsmodels R-style formulas.

    ANTsR function: `ilr`

    Arguments
    ---------

    data_frame: This data frame contains all relevant predictors except for
        the matrices associated with the image variables.  One should convert
        any categorical predictors ahead of time using `pd.get_dummies`.

    voxmats: The named list of matrices that contains the changing
        predictors.

    ilr_formula: This is a character string that defines a valid regression
        formula in the R-style.

    verbose:  will print a little bit of diagnostic information that allows
        a degree of model checking

    Returns
    -------

    A list of different matrices that contain names derived from the
    formula and the coefficients of the regression model.  The size of
    the output values ( p-values, t-values, parameter values ) will match
    the input matrix and, as such, can be converted to an image via `make_image`

    Example
    -------

    >>> nsub = 20
    >>> mu, sigma = 0, 1
    >>> outcome = np.random.normal( mu, sigma, nsub )
    >>> covar = np.random.normal( mu, sigma, nsub )
    >>> mat = np.random.normal( mu, sigma, (nsub, 500 ) )
    >>> mat2 = np.random.normal( mu, sigma, (nsub, 500 ) )
    >>> data = {'covar':covar,'outcome':outcome}
    >>> df = pd.DataFrame( data )
    >>> vlist = { "mat1": mat, "mat2": mat2 }
    >>> myform = " outcome ~ covar * mat1 "
    >>> result = ilr( df, vlist, myform)
    >>> myform = " mat2 ~ covar + mat1 "
    >>> result = ants.ilr( df, vlist, myform)

    """

    nvoxmats = len( voxmats )
    if nvoxmats < 1 :
        raise ValueError('Pass at least one matrix to voxmats list')
    keylist = list(voxmats.keys())
    firstmat = keylist[0]
    voxshape = voxmats[firstmat].shape
    nvox = voxshape[1]
    nmats = len( keylist )
    for k in keylist:
        if voxmats[firstmat].shape != voxmats[k].shape:
            raise ValueError('Matrices must have same number of rows (samples)')

    # test voxel
    vox = 0
    nrows = data_frame.shape[0]
    data_frame_vox = data_frame.copy()
    for k in range( nmats ):
        data = {keylist[k]: np.random.normal(0,1,nrows) }
        temp = pd.DataFrame( data )
        data_frame_vox = pd.concat([data_frame_vox.reset_index(drop=True),temp], axis=1 )
    mod = smf.ols(formula=ilr_formula, data=data_frame_vox )
    res = mod.fit()
    modelNames = res.model.exog_names
    if verbose:
        print( data_frame_vox )
        print(res.summary())
    nOutcomes = len( modelNames )
    tValsOut = list()
    pValsOut = list()
    bValsOut = list()
    for k in range( len( modelNames ) ):
        bValsOut.append( np.zeros( nvox ) )
        pValsOut.append( np.zeros( nvox ) )
        tValsOut.append( np.zeros( nvox ) )

    data_frame_vox = data_frame.copy()
    for v in range( nmats ):
        data = {keylist[v]: voxmats[keylist[v]][:,k] }
        temp = pd.DataFrame( data )
        data_frame_vox = pd.concat([data_frame_vox.reset_index(drop=True),temp], axis=1 )
    for k in range( nvox ):
        # first get the correct data frame
        for v in range( nmats ):
            data_frame_vox[ keylist[v] ] = voxmats[keylist[v]][:,k]
        # then get the local model results
        mod = smf.ols(formula=ilr_formula, data=data_frame_vox )
        res = mod.fit()
        tvals = res.tvalues
        pvals = res.pvalues
        bvals = res.params
        for v in range( len( modelNames ) ):
            bValsOut[v][k] = bvals[v]
            pValsOut[v][k] = pvals[v]
            tValsOut[v][k] = tvals[v]

    bValsOutDict = { }
    tValsOutDict = { }
    pValsOutDict = { }
    for v in range( len( modelNames ) ):
        bValsOutDict[ 'coef_' + modelNames[v] ] = bValsOut[v]
        tValsOutDict[ 'tval_' + modelNames[v] ] = tValsOut[v]
        pValsOutDict[ 'pval_' + modelNames[v] ] = pValsOut[v]

    return {
        'modelNames': modelNames,
        'coefficientValues': bValsOutDict,
        'pValues': pValsOutDict,
        'tValues': tValsOutDict }



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

def bandpass_filter_matrix( matrix,
    tr=1, lowf=0.01, highf=0.1, order = 3):
    """
    Bandpass filter the input time series image

    ANTsR function: `frequencyFilterfMRI`

    Arguments
    ---------

    image: input time series image

    tr:    sampling time interval (inverse of sampling rate)

    lowf:  low frequency cutoff

    highf: high frequency cutoff

    order: order of the butterworth filter run using `filtfilt`

    Returns
    -------
    filtered matrix

    Example
    -------

    >>> import numpy as np
    >>> import ants
    >>> import matplotlib.pyplot as plt
    >>> brainSignal = np.random.randn( 400, 1000 )
    >>> tr = 1
    >>> filtered = ants.bandpass_filter_matrix( brainSignal, tr = tr )
    >>> nsamples = brainSignal.shape[0]
    >>> t = np.linspace(0, tr*nsamples, nsamples, endpoint=False)
    >>> k = 20
    >>> plt.plot(t, brainSignal[:,k], label='Noisy signal')
    >>> plt.plot(t, filtered[:,k], label='Filtered signal')
    >>> plt.xlabel('time (seconds)')
    >>> plt.grid(True)
    >>> plt.axis('tight')
    >>> plt.legend(loc='upper left')
    >>> plt.show()
    """
    from scipy.signal import butter, filtfilt

    def butter_bandpass(lowcut, highcut, fs, order ):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order ):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    fs = 1/tr   # sampling rate based on tr
    nsamples = matrix.shape[0]
    ncolumns = matrix.shape[1]
    matrixOut = matrix.copy()
    for k in range( ncolumns ):
        matrixOut[:,k] = butter_bandpass_filter(
            matrix[:,k], lowf, highf, fs, order=order )
    return matrixOut


def compcor( boldImage, ncompcor=4, quantile=0.975, mask=None, filter_type=False, degree=2 ):
    """
    Compute noise components from the input image

    ANTsR function: `compcor`

    this is adapted from nipy code https://github.com/nipy/nipype/blob/e29ac95fc0fc00fedbcaa0adaf29d5878408ca7c/nipype/algorithms/confounds.py

    Arguments
    ---------

    boldImage: input time series image

    ncompcor:  number of noise components to return

    quantile:  quantile defining high-variance

    mask:      mask defining brain or specific tissues

    filter_type: type off filter to apply to time series before computing
                 noise components.

        'polynomial' - Legendre polynomial basis
        False - None (mean-removal only)

    degree: order of polynomial used to remove trends from the timeseries

    Returns
    -------
    dictionary containing:

        components: a numpy array

        basis: a numpy array containing the (non-constant) filter regressors

    Example
    -------
    >>> cc = ants.compcor( ants.image_read(ants.get_ants_data("ch2")) )

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
