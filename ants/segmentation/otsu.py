
 
__all__ = ['otsu_segmentation']

from .. import utils

def otsu_segmentation(image, k, mask=None):
    """
    Otsu image segmentation

    This is a very fast segmentation algorithm good for quick explortation, 
    but does not return probability maps.

    ANTsR function: `thresholdImage(image, 'Otsu', k)`
    
    Arguments
    ---------
    image : ANTsImage 
        input image
    
    k : integer
        integer number of classes. Note that a background class will 
        be added to this, so the resulting segmentation will 
        have k+1 unique values.
    
    mask : ANTsImage 
        segment inside this mask

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> seg = mni.otsu_segmentation(k=3) #0=bg,1=csf,2=gm,3=wm
    """
    if mask is not None:
        image = image.mask_image(mask)

    seg = image.threshold_image('Otsu', k)
    return seg



