__all__ = ["regression_match_image"]

import ants

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import numpy as np

def regression_match_image(source_image,
                           reference_image,
                           mask=None,
                           poly_order=1,
                           truncate=True):
    """
    Image intensity normalization using linear regression.

    Arguments
    ---------
    source_image : ANTsImage
        Image whose intensities are matched to the reference
        image.

    reference_image : ANTsImage
        Defines the reference intensity function.

    poly_order : integer
        Polynomial order of fit.  Default is 1 (linear fit).

    mask : ANTsImage
        Defines voxels for regression modeling.

    truncate : boolean
        Turns on/off the clipping of intensities.

    Returns
    -------
    ANTs image (i.e., source_image) matched to the (reference_image).

    Example
    -------
    >>> import ants
    >>> source_image = ants.image_read(ants.get_ants_data('r16'))
    >>> reference_image = ants.image_read(ants.get_ants_data('r64'))
    >>> matched_image = regression_match_image(source_image, reference_image)
    """

    if source_image.shape != reference_image.shape:
        raise ValueError("Images do not have the same dimension.")

    if mask is None:
        source_intensities = np.expand_dims((source_image.numpy()).flatten(), axis=1)
        reference_intensities = np.expand_dims((reference_image.numpy()).flatten(), axis=1)
    else:
        mask_intensities = (mask.numpy()).flatten()
        source_intensities = np.expand_dims(source_image.numpy().flatten()[np.where(mask_intensities != 0)], axis=1)
        reference_intensities = np.expand_dims(reference_image.numpy().flatten()[np.where(mask_intensities != 0)], axis=1)

    poly_features = PolynomialFeatures(degree=poly_order)
    source_intensities_poly = poly_features.fit_transform(source_intensities)
    model = LinearRegression()
    model.fit(source_intensities_poly, reference_intensities)
    if mask is not None:
        source_intensities_poly = poly_features.fit_transform(np.expand_dims((source_image.numpy()).flatten(), axis=1))
    matched_source_intensities = model.predict(source_intensities_poly)

    if truncate == True:
        min_reference_value = reference_intensities.min()
        max_reference_value = reference_intensities.max()
        matched_source_intensities[matched_source_intensities < min_reference_value] = min_reference_value
        matched_source_intensities[matched_source_intensities > max_reference_value] = max_reference_value

    matched_source_image = ants.make_image(source_image.shape, matched_source_intensities)
    matched_source_image = ants.copy_image_info(source_image,  matched_source_image)

    return(matched_source_image)
