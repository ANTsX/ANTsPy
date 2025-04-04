import numpy as np
import ants


def from_sitk(sitk_image: "SimpleITK.Image") -> ants.ANTsImage:
    """
    Converts a given SimpleITK image into an ANTsPy image

    Parameters
    ----------
        img: SimpleITK.Image

    Returns
    -------
        ants_image: ANTsImage
    """
    import SimpleITK as sitk

    ndim = sitk_image.GetDimension()

    if ndim < 3:
        print("Dimensionality is less than 3.")
        return None

    direction = np.asarray(sitk_image.GetDirection()).reshape((3, 3))
    spacing = list(sitk_image.GetSpacing())
    origin = list(sitk_image.GetOrigin())

    data = sitk.GetArrayViewFromImage(sitk_image)

    ants_img: ants.ANTsImage = ants.from_numpy(
        data=data.ravel(order="F").reshape(data.shape[::-1]),
        origin=origin,
        spacing=spacing,
        direction=direction,
    )

    return ants_img


def to_sitk(ants_image: ants.ANTsImage) -> "SimpleITK.Image":
    """
    Converts a given ANTsPy image into an SimpleITK image

    Parameters
    ----------
        ants_image: ANTsImage

    Returns
    -------
        img: SimpleITK.Image
    """

    import SimpleITK as sitk

    data = ants_image.view()
    shape = ants_image.shape

    sitk_img = sitk.GetImageFromArray(data.ravel(order="F").reshape(shape[::-1]))
    sitk_img.SetOrigin(ants_image.origin)
    sitk_img.SetSpacing(ants_image.spacing)
    sitk_img.SetDirection(ants_image.direction.flatten())
    return sitk_img
