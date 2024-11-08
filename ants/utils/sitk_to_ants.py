import numpy as np
import ants

def image_to_ants(sitk_image: "SimpleITK.Image") -> ants.ANTsImage:
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