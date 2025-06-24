__all__ = ['from_sitk', 'to_sitk']

import numpy as np
import ants

def from_sitk(sitk_image: "SimpleITK.Image") -> ants.ANTsImage:
    """
    Converts a SimpleITK image into an ANTsPy image. Requires SimpleITK.

    Parameters
    ----------
    sitk_image : SimpleITK.Image
        The SimpleITK image to convert.

    Returns
    -------
    ants_image : ANTsImage
        The converted ANTsPy image.

    Raises
    ------
    ValueError
        If the image dimensionality is unsupported.

    Examples
    --------
    >>> import SimpleITK as sitk
    >>> import ants
    >>> img = sitk.ReadImage('brain.nii.gz')
    >>> ants_img = from_sitk(img)

    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError(
            "SimpleITK is required to convert to or from SimpleITK images. Install SimpleITK to use this function."
        )

    ndim = sitk_image.GetDimension()
    if ndim not in (2, 3, 4):
        raise ValueError(f"Unsupported SimpleITK image dimension: {ndim}")

    is_vector = sitk_image.GetNumberOfComponentsPerPixel() > 1

    data = sitk.GetArrayViewFromImage(sitk_image)

    if is_vector:
        # Vector images: SimpleITK gives (z,y,x,components) or (y,x,components)
        spatial_shape = data.shape[:-1]  # drop components
    else:
        spatial_shape = data.shape

    image_dimension = len(spatial_shape)

    if image_dimension == 2:
        direction = np.asarray(sitk_image.GetDirection()).reshape((2, 2))
        spacing = list(sitk_image.GetSpacing())
        origin = list(sitk_image.GetOrigin())
    elif image_dimension == 3:
        direction = np.asarray(sitk_image.GetDirection()).reshape((3, 3))
        spacing = list(sitk_image.GetSpacing())
        origin = list(sitk_image.GetOrigin())
    elif image_dimension == 4:
        direction = np.asarray(sitk_image.GetDirection()).reshape((4, 4))
        spacing = list(sitk_image.GetSpacing())
        origin = list(sitk_image.GetOrigin())
    else:
        raise ValueError(f"Unsupported image dimension: {image_dimension}")

    # Reshape the array properly for ANTsPy
    if is_vector:
        data_reshaped = data.transpose(list(range(image_dimension - 1, -1, -1)) + [image_dimension])
    else:
        data_reshaped = data.transpose(list(range(image_dimension - 1, -1, -1)))

    ants_img: ants.ANTsImage = ants.from_numpy(
        data=data_reshaped,
        origin=origin,
        spacing=spacing,
        direction=direction,
        has_components=is_vector
    )

    return ants_img


def to_sitk(ants_image: ants.ANTsImage) -> "SimpleITK.Image":
    """
    Converts an ANTsPy image into a SimpleITK image. Requires SimpleITK.

    Parameters
    ----------
    ants_image : ANTsImage
        The ANTsPy image to convert.

    Returns
    -------
    sitk_image : SimpleITK.Image
        The converted SimpleITK image.

    Raises
    ------
    ValueError
        If the image dimensionality is unsupported.

    Examples
    --------
    >>> import ants
    >>> ants_img = ants.image_read('brain.nii.gz')
    >>> sitk_img = ants.to_sitk(ants_img)

    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError(
            "SimpleITK is required to convert to or from SimpleITK images. Install SimpleITK to use this function."
        )

    if ants_image.dimension not in (2, 3, 4):
        raise ValueError(f"Unsupported ANTsPy image dimension: {ants_image.dimension}")

    data = ants_image.view()
    shape = ants_image.shape
    components = ants_image.components

    is_vector = components > 1

    # Reshape for SimpleITK
    if is_vector:
        # data shape (x,y,z,components) -> sitk expects (z,y,x,components)
        array = data.ravel(order="F").reshape(shape[::-1] + (components,))
    else:
        # data shape (x,y,z) -> sitk expects (z,y,x)
        array = data.ravel(order="F").reshape(shape[::-1])

    sitk_img = sitk.GetImageFromArray(array, isVector=is_vector)

    sitk_img.SetOrigin(ants_image.origin)
    sitk_img.SetSpacing(ants_image.spacing)
    sitk_img.SetDirection(ants_image.direction.flatten())

    return sitk_img
