"""
Short-name nibabel conversion helpers: ``ants.to_nibabel`` / ``ants.from_nibabel``.

This module restores the public names requested in ANTsPy issue #693
(``ants.to_nibabel`` and ``ants.from_nibabel`` were missing from the 0.5.3
API). The names are thin wrappers over the ``*_nifti`` implementations in
:mod:`ants.utils.nibabel_nifti_to_ants`, which do the conversion directly
from the image geometry instead of round-tripping through a temporary file
on disk the way the pre-0.5.3 versions did.

nibabel stays an *optional* dependency: it is only imported lazily, inside
:func:`ants.utils.nibabel_nifti_to_ants.to_nibabel_nifti`, so importing
``ants`` -- and calling :func:`from_nibabel` on any object exposing the
nibabel image interface (``.affine``, ``.header``, ``.get_fdata()``) -- works
with no nibabel installed.

Affine conventions (the part that historically went wrong, see #52, #64)
------------------------------------------------------------------------
* nibabel / NIfTI work in RAS+ world coordinates; ANTsPy / ITK work in
  LPS+ coordinates. Converting flips the sign of the first two world axes,
  i.e. the affine is left-multiplied by ``diag(-1, -1, 1, 1)`` in both
  directions.
* ``from_nibabel`` reads ``nib_image.affine``. nibabel builds that affine
  from the header as: the **sform** when ``sform_code > 0``, otherwise the
  **qform** (quaternion + zooms + offsets). So when both are present the
  sform wins, matching nibabel's own documented rule.
* ITK can only represent a rigid (+ scaling) direction matrix, so a
  sheared sform is orthogonalized ("desheared") before the direction is
  extracted; if deshearing would move an axis by more than
  ``max_angle_deviation`` degrees a ``ValueError`` is raised. In that
  case -- or whenever the sform is non-rigid -- the result can diverge
  from what nibabel itself would report, because nibabel keeps the
  sheared sform as-is. A rigid sform (``sform_code > 0``) round-trips
  exactly.
* ``to_nibabel`` writes the ANTsPy geometry as the NIfTI **sform**
  (``sform_code`` 2, 'aligned') and nibabel derives a matching qform from it, so
  sform and qform agree on the output image. Spatial/time units are set
  to mm / sec.
"""

__all__ = ["to_nibabel", "from_nibabel"]

from .nibabel_nifti_to_ants import from_nibabel_nifti, to_nibabel_nifti


def to_nibabel(ants_image, header=None):
    """
    Convert an ANTsImage to a nibabel NIfTI image.

    This is the short public name for
    :func:`ants.to_nibabel_nifti` (see there for the full description).
    It restores the ``ants.to_nibabel`` entry point requested in issue
    #693, which was missing from the 0.5.3 API.

    The ANTsPy (ITK, LPS+) geometry is converted to nibabel (NIfTI,
    RAS+) geometry by flipping the sign of the first two world axes,
    and written out as the NIfTI sform (``sform_code`` 2, 'aligned') with a
    matching qform; spatial/time units are set to mm / sec. The data
    array itself is passed through unchanged (multi-component images
    are expanded to a 5D array, as nibabel expects components in the
    5th dimension).

    Parameters
    ----------
    ants_image : ANTsImage
        The input ANTsPy image. Should be a 3D, 4D (time series) or
        5D (multi-component) image.
    header : nibabel Nifti1Header, optional
        An optional Nifti1Header to use for the output NIfTI image.
        If the transform of this header differs from the input image,
        the header qform is removed, the input image transform is
        used as the sform, and the sform code is set to 'aligned'.

    Returns
    -------
    nib_image : nibabel Nifti1Image

    Raises
    ------
    ImportError
        If nibabel is not installed. nibabel is an optional
        dependency: install it with ``pip install nibabel``.
    ValueError
        If the image dimensionality is not 3, 4 or 5.

    Examples
    --------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('mni'))
    >>> nib_img = ants.to_nibabel(img)
    """
    return to_nibabel_nifti(ants_image, header=header)


def from_nibabel(
    nib_image, deshear_threshold=1e-6, max_angle_deviation=0.5, pixeltype="float"
):
    """
    Convert a nibabel NIfTI image to an ANTsImage.

    This is the short public name for
    :func:`ants.from_nibabel_nifti` (see there for the full
    description). It restores the ``ants.from_nibabel`` entry point
    requested in issue #693, which was missing from the 0.5.3 API.

    The nibabel (NIfTI, RAS+) affine is converted to ANTsPy (ITK,
    LPS+) geometry by flipping the sign of the first two world axes.
    Which affine is used follows nibabel's own rule: ``nib_image.affine``
    is the header **sform** when ``sform_code > 0``, otherwise the
    **qform**. Because ITK cannot represent shear, a sheared sform is
    orthogonalized first (see ``deshear_threshold`` /
    ``max_angle_deviation``); a rigid sform (``sform_code > 0``)
    converts exactly.

    Unlike :func:`to_nibabel`, this direction does not import nibabel
    at all: any object exposing the nibabel image interface
    (``.affine``, ``.header``, ``.get_fdata()``) works.

    Parameters
    ----------
    nib_image : nibabel NIfTI image
        The input image. Must be 3D, 4D (time series) or 5D
        (multi-component). Spatial units should be mm and time units
        seconds; 'unknown' units are assumed to be mm / sec.
    deshear_threshold : float
        Threshold for action to remove the shear component of the
        affine transform. If shear is above this threshold, it is
        removed. Otherwise, the affine is used as-is.
    max_angle_deviation : float
        Maximum angle deviation in degrees for directions after
        deshearing the affine. If the desheared directions deviate
        from the originals by more than this value (in degrees), a
        ``ValueError`` is raised.
    pixeltype : str
        Pixel type for the output ANTsPy image. One of "uint8",
        "uint32", "float32", "float64".

    Returns
    -------
    ants_image : ANTsImage

    Raises
    ------
    ValueError
        If the image dimensionality is unsupported, if the spatial /
        time units are not mm / sec, if the spacing extracted from
        the affine does not match the header zooms, or if deshearing
        would move an axis by more than ``max_angle_deviation``.

    Examples
    --------
    >>> import ants
    >>> import nibabel as nib
    >>> nib_img = nib.load('brain.nii.gz')
    >>> ants_img = ants.from_nibabel(nib_img)
    """
    return from_nibabel_nifti(
        nib_image,
        deshear_threshold=deshear_threshold,
        max_angle_deviation=max_angle_deviation,
        pixeltype=pixeltype,
    )
