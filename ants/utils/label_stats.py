
__all__ = ['label_stats']

from .. import lib

_label_stats_dict = {
    2: lib.labelStats2D,
    3: lib.labelStats3D,
    4: lib.labelStats4D
}

def label_stats(image, label_image):
    image_float = image.clone('float')
    label_image_int = label_image.clone('unsigned int')
    label_stats_fn = _label_stats_dict[image.dimension]

    df = label_stats_fn(image_float._img, label_image_int._img)
    #df = df[order(df$LabelValue), ]
    return df
