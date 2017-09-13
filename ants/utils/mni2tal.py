
__all__ = ['mni2tal']

def mni2tal(xin):
    """
    mni2tal for converting from ch2/mni space to tal - very approximate.

    This is a standard approach but it's not very accurate.

    ANTsR function: `mni2tal`

    Arguments
    ---------
    xin : tuple
        point in mni152 space.

    Returns
    -------
    tuple

    Example
    -------
    >>> import ants
    >>> ants.mni2tal( (10,12,14) )

    References
    ----------
    http://bioimagesuite.yale.edu/mni2tal/501_95733_More\%20Accurate\%20Talairach\%20Coordinates\%20SLIDES.pdf
    http://imaging.mrc-cbu.cam.ac.uk/imaging/MniTalairach
    """
    if (not isinstance(xin, (tuple,list))) or (len(xin) != 3):
        raise ValueError('xin must be tuple/list with 3 coordinates')

    x = list(xin)
    # The input image is in RAS coordinates but we use ITK which returns LPS
    # coordinates.  So we need to flip the coordinates such that L => R and P => A to
    # get RAS (MNI) coordinates
    x[0] = x[0] * (-1)  # flip X
    x[1] = x[1] * (-1)  # flip Y
    
    xout = x
    
    if (x[2] >= 0):
        xout[0] = x[0] * 0.99
        xout[1] = x[1] * 0.9688 + 0.046 * x[2]
        xout[2] = x[1] * (-0.0485) + 0.9189 * x[2]
    
    if (x[2] < 0):
        xout[0] = x[0] * 0.99
        xout[1] = x[1] * 0.9688 + 0.042 * x[2]
        xout[2] = x[1] * (-0.0485) + 0.839 * x[2]
    
    return(xout)






