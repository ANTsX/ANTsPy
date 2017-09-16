Core
===================================
.. automodule:: ants

Images
----------------------------------

ANTsImage
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ants.core.ants_image.ANTsImage
   :members:

ANTsImage IO
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: image_clone
.. autofunction:: image_header_info
.. autofunction:: image_read
.. autofunction:: image_write
.. autofunction:: make_image
.. autofunction:: from_numpy
.. autofunction:: matrix_to_images
.. autofunction:: images_from_matrix
.. autofunction:: image_list_to_matrix
.. autofunction:: images_to_matrix
.. autofunction:: matrix_from_images

Transforms
----------------------------------

ANTsTransform
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ants.core.ants_transform.ANTsTransform
   :members:

ANTsTransform IO
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: create_ants_transform
.. autofunction:: new_ants_transform
.. autofunction:: read_transform
.. autofunction:: write_transform
.. autofunction:: transform_from_displacement_field

Metrics
----------------------------------

ANTsMetric
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ants.core.ants_metric.ANTsImageToImageMetric
   :members:

ANTsMetric IO
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: new_ants_metric
.. autofunction:: create_ants_metric
.. autofunction:: supported_metrics


