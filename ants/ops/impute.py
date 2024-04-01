

__all__ = ['impute']

import numpy as np

try:
    from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, IterativeSVD
    has_fancyimpute = True
except:
    has_fancyimpute = False


def impute(data, method='mean', value=None, nan_value=np.nan):
    """
    Impute missing values on a numpy ndarray in a column-wise manner.
    
    ANTsR function: `antsrimpute`

    Arguments
    ---------
    data : numpy.ndarray
        data to impute

    method : string or float
        type of imputation method to use
        Options:
            mean
            median
            constant
            KNN
            BiScaler
            NuclearNormMinimization
            SoftImpute
            IterativeSVD

    value : scalar (optional)
        optional arguments for different methods
        if method == 'constant'
            constant value
        if method == 'KNN'
            number of nearest neighbors to use

    nan_value : scalar
        value which is interpreted as a missing value

    Returns
    -------
    ndarray if ndarray was given
    OR
    pd.DataFrame if pd.DataFrame was given

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> data = np.random.randn(4,10)
    >>> data[2,3] = np.nan
    >>> data[3,5] = np.nan
    >>> data_imputed = ants.impute(data, 'mean')

    Details
    -------
    KNN: Nearest neighbor imputations which weights samples using the mean squared 
            difference on features for which two rows both have observed data.

    SoftImpute: Matrix completion by iterative soft thresholding of SVD 
                decompositions. Inspired by the softImpute package for R, which 
                is based on Spectral Regularization Algorithms for Learning 
                Large Incomplete Matrices by Mazumder et. al.

    IterativeSVD: Matrix completion by iterative low-rank SVD decomposition.
                    Should be similar to SVDimpute from Missing value estimation 
                    methods for DNA microarrays by Troyanskaya et. al.

    MICE: Reimplementation of Multiple Imputation by Chained Equations.

    MatrixFactorization: Direct factorization of the incomplete matrix into 
                        low-rank U and V, with an L1 sparsity penalty on the elements 
                        of U and an L2 penalty on the elements of V. 
                        Solved by gradient descent.

    NuclearNormMinimization: Simple implementation of Exact Matrix Completion 
                            via Convex Optimization by Emmanuel Candes and Benjamin 
                            Recht using cvxpy. Too slow for large matrices.

    BiScaler: Iterative estimation of row/column means and standard deviations 
                to get doubly normalized matrix. Not guaranteed to converge but 
                works well in practice. Taken from Matrix Completion and 
                Low-Rank SVD via Fast Alternating Least Squares.
    """
    _fancyimpute_options = {'KNN', 'BiScaler', 'NuclearNormMinimization', 'SoftImpute', 'IterativeSVD'}
    if (not has_fancyimpute) and (method in _fancyimpute_options):
        raise ValueError('You must install `fancyimpute` (pip install fancyimpute) to use this method')

    _base_options = {'mean', 'median', 'constant'}
    if (method not in _base_options) and (method not in _fancyimpute_options) and (not isinstance(method, (int,float))):
        raise ValueError('method not understood.. Use `mean`, `median`, a scalar, or an option from `fancyimpute`')

    X_incomplete = data.copy()

    if method == 'KNN':
        if value is None:
            value = 3
        X_filled = KNN(k=value, verbose=False).complete(X_incomplete)

    elif method == 'BiScaler':
        X_filled = BiScaler(verbose=False).fit_transform(X_incomplete)

    elif method == 'SoftImpute':
        X_filled = SoftImpute(verbose=False).complete(X_incomplete)

    elif method == 'IterativeSVD':
        if value is None:
            rank = min(10, X_incomplete.shape[0]-2)
        else:
            rank = value
        X_filled = IterativeSVD(rank=rank, verbose=False).complete(X_incomplete)

    elif method == 'mean':
        col_means = np.nanmean(X_incomplete, axis=0)
        for i in range(X_incomplete.shape[1]):
            X_incomplete[:,i][np.isnan(X_incomplete[:,i])] = col_means[i]
        X_filled = X_incomplete

    elif method == 'median':
        col_means = np.nanmean(X_incomplete, axis=0)
        for i in range(X_incomplete.shape[1]):
            X_incomplete[:,i][np.isnan(X_incomplete[:,i])] = col_means[i]
        X_filled = X_incomplete

    elif method == 'constant':
        if value is None:
            raise ValueError('Must give `value` argument if method == constant')
        X_incomplete[np.isnan(X_incomplete)] = value
        X_filled = X_incomplete

    return X_filled






