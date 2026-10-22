NumPy image metadata sidecars
============================

For ``.npy`` images, metadata is read from the adjacent JSON file formed by
replacing only the final ``.npy`` suffix with ``.json``. Without that sidecar,
images use default origin, spacing and direction.

Older versions replaced every ``.npy`` occurrence in the path. For an existing
image such as ``directory.npy/image.npy.copy.npy``, move its metadata from
``directory.json/image.json.copy.json`` to
``directory.npy/image.npy.copy.json``. Check that the metadata belongs to the
image and do not overwrite an existing sidecar. Legacy paths are not searched.
