__all__ = [
    "deshear_sform",
    "get_nifti_sform_shear",
    "get_nifti_qform_spatial_info",
    "get_nifti_sform_spatial_info",
    "get_nifti_spatial_transform_from_metadata",
    "set_nifti_spatial_transform_from_metadata"
]

import numpy as np
import ants
import warnings

def _as_affine_4x4_from_srow(md):
    """Build 4x4 sform affine from srow_x/y/z (strings of 4 floats) or return None."""
    rx, ry, rz = (md.get("srow_x"), md.get("srow_y"), md.get("srow_z"))
    if rx is None or ry is None or rz is None:
        return None
    def _parse_row(r):
        return [float(x) for x in r.split()]
    A = np.eye(4, dtype=float)
    A[0, :] = _parse_row(rx)
    A[1, :] = _parse_row(ry)
    A[2, :] = _parse_row(rz)
    return A


def _as_affine_4x4_from_qto(md):
    """Read qform as 4x4 from 'qto_xyz' (already a 4x4 list) or return None."""
    q = md.get("qto_xyz")
    if q is None:
        return None
    A = np.array(q, dtype=float)
    if A.shape == (4, 4):
        return A
    return None


def _ras_to_lps_affine(A):
    """Convert an index->RAS affine to index->LPS by left-multiplying diag([-1,-1,1,1])."""
    L = np.diag([-1.0, -1.0, 1.0, 1.0])
    return L @ A


def _spacing_and_dirs_from_affine(A3):
    """
    From the 3x3 part of an index->world affine, return:
      spacing: norms of columns
      dirs:    unit direction columns (3x3)
    """
    sx = np.linalg.norm(A3[:, 0])
    sy = np.linalg.norm(A3[:, 1])
    sz = np.linalg.norm(A3[:, 2])
    # Protect against zero
    sx = sx if sx > 0 else 1.0
    sy = sy if sy > 0 else 1.0
    sz = sz if sz > 0 else 1.0
    D = np.column_stack([A3[:, 0] / sx, A3[:, 1] / sy, A3[:, 2] / sz])
    return [sx, sy, sz], D


def _shear_components_from_dirs(D):
    """
    Shear components as cosine between the (supposed-to-be) orthogonal direction axes
    [shear_xy, shear_xz, shear_yz]. Perfect orthonormal -> [0,0,0].
    """
    ux, uy, uz = D[:, 0], D[:, 1], D[:, 2]
    return [float(np.dot(ux, uy)), float(np.dot(ux, uz)), float(np.dot(uy, uz))]



def _deshear_affine(affine, eps=1e-12):
    """
    Remove shear from a 4x4 index->world affine by orthogonalizing the 3x3 part
    with a column-preserving QR decomposition.

    - Preserves axis order (no permutations).
    - Enforces a proper rotation (no reflections) by making the diagonal of R positive.
    - Keeps per-axis scales from the diagonal of R.
    - Leaves translation unchanged.

    Parameters
    ----------
    affine : (4,4) array
        Input affine.
    eps : float
        Small threshold to avoid division / sign issues when a scale is ~0.

    Returns
    -------
    (4,4) array
        Affine with zero shear.
    """
    A = np.array(affine, dtype=float, copy=True)
    if A.shape != (4, 4):
        raise ValueError("affine must be 4x4")

    R3 = A[:3, :3]

    # Column-preserving Gram–Schmidt via QR
    # R3 = Q * R, with Q orthogonal, R upper-triangular
    Q, R = np.linalg.qr(R3)

    # Force positive scales on the diagonal of R (avoid reflections / sign flips)
    d = np.sign(np.diag(R))
    d[np.abs(np.diag(R)) < eps] = 1.0  # don't flip near-zero
    D = np.diag(d)

    Q = Q @ D            # flip columns of Q as needed
    R = D @ R            # make diag(R) nonnegative

    # Extract per-axis scales from the diagonal; drop shear (off-diagonals)
    scales = np.diag(R).copy()

    # Guard against zeros
    scales[np.abs(scales) < eps] = 1.0

    # Recompose: orthogonal dirs (Q) with per-axis scales (no shear)
    A_new = A.copy()
    A_new[:3, :3] = Q @ np.diag(scales)

    return A_new


def _angle_deg(u, v):
    """Angle between two vectors (degrees)."""
    u = np.array(u, float); v = np.array(v, float)
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    c = np.dot(u/nu, v/nv)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _pixdims_from_metadata(md):
    # NIfTI uses pixdim[1], pixdim[2], pixdim[3] (strings in your ITK dict)
    keys = [f"pixdim[{i}]" for i in (1, 2, 3)]
    vals = []
    for k in keys:
        v = md.get(k)
        if v is None:
            return None
        try:
            vals.append(float(v))
        except Exception:
            return None
    return vals

def _spacing_matches(a, b, rtol=1e-3, atol=1e-6):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.allclose(a, b, rtol=rtol, atol=atol)


# --- public functions ---

def get_nifti_sform_shear(metadata):
    """
    Get the shear parameters from the sform matrix in metadata dictionary.

    Returns
    -------
    [shear_xy, shear_xz, shear_yz]  (cosines between axes; 0 means orthogonal)

    Example
    -------
    >>> import ants
    >>> metadata = ants.read_image_metadata( ants.get_ants_data('mni') )
    >>> ants.get_nifti_sform_shear(metadata)
    """
    A = _as_affine_4x4_from_srow(metadata)
    if A is None:
        raise ValueError("No sform found (srow_x/y/z missing)")
    _, D = _spacing_and_dirs_from_affine(A[:3, :3])
    return _shear_components_from_dirs(D)


def deshear_affine_transform_matrix(A, deshear_threshold=1e-6, max_angle_deviation=0.5):
    """
    Deshear an affine transform by removing shear components.

    Arguments
    ---------
    affine : numpy.ndarray array
        The 4x4 affine matrix to deshear.
    deshear_threshold : float
        Threshold for taking action to deshear the sform. Shear below this value is considered negligible, and the original
        affine is returned.
    max_angle_deviation : float
        Maximum angle deviation in degrees for directions after deshearing sform. If the desheared directions deviate from the
        originals by more than this value (in degrees), a ValueError is raised.

    Returns
    -------
    numpy.ndarray array
        The desheared affine matrix.

    """
    spacing, D = _spacing_and_dirs_from_affine(A[:3, :3])
    shear_xy, shear_xz, shear_yz = _shear_components_from_dirs(D)
    shear_max = max(abs(shear_xy), abs(shear_xz), abs(shear_yz))

    if shear_max <= deshear_threshold:
        # Already fine
        return A.copy()

    Q = _deshear_affine(A)

    dev = max(
        _angle_deg(A[:, 0], Q[:, 0]),
        _angle_deg(A[:, 1], Q[:, 1]),
        _angle_deg(A[:, 2], Q[:, 2]),
    )
    if dev > max_angle_deviation:
        raise ValueError(
            f"Desheared directions deviate by {dev:.3f} deg (> max allowed {max_angle_deviation} deg). "
            "This likely indicates significant obliquity/shear"
        )

    # Rebuild affine with same translation
    return Q


def deshear_nifti_sform(metadata, deshear_threshold=1e-6, max_angle_deviation=0.5):
    """
    Deshear an sform matrix in the provided metadata dict.
    Returns a new 4x4 affine (index->RAS) with shear removed (direction orthonormal),
    preserving spacings and translation. Raises ValueError if the orthonormalized
    directions deviate from the originals by more than max_angle_deviation degrees.

    Arguments
    ---------
    metadata : dict
        The NIfTI header metadata dictionary.
    deshear_threshold : float
        Shear threshold for deshearing sform. Shear below this value is considered negligible, and the original sform is
        returned.
    max_angle_deviation : float
        Maximum angle deviation for directions after deshearing sform. If the desheared directions deviate from the original
        directions by more than this value (in degrees), a ValueError is raised.

    Returns
    -------
    numpy.ndarray array
        The desheared affine matrix.

    Example
    -------
    >>> import ants
    >>> metadata = ants.read_image_metadata( ants.get_ants_data('mni') )
    >>> A = ants.deshear_nifti_sform(metadata)
    """
    A = _as_affine_4x4_from_srow(metadata)
    if A is None:
        raise ValueError("No sform found (srow_x/y/z missing)")

    return deshear_affine_transform_matrix(A, deshear_threshold=deshear_threshold, max_angle_deviation=max_angle_deviation)


def get_nifti_sform_spatial_info(metadata, deshear_threshold=1e-6, max_angle_deviation=0.5):
    """
    Extract sform-derived spacing, origin, direction.

    Note: output is in ITK LPS coordinates.

    Arguments
    ---------
    metadata : dict
        The NIfTI header metadata dictionary.
    deshear_threshold : float
        Shear threshold for deshearing sform, if the shear is beneath this value, the sform is not modified.
    max_angle_deviation : float
        Maximum angle deviation for directions after deshearing sform. If the desheared directions deviate from the original
        directions by more than this value (in degrees), deshearing fails and the return value is None.

    Returns
    -------
    dict with keys:
      pixdim_spacing = [sx, sy, sz]  (from pixdim[1..3])
      transform_spacing : [sx, sy, sz]
      origin  : [ox, oy, oz]
      direction : 3x3 list (direction cosines)
      desheared : bool
      original_shear : [shear_xy, shear_xz, shear_yz]

    Returns None if no sform is present or if deshearing fails.

    Example
    -------
    >>> import ants
    >>> metadata = ants.read_image_metadata( ants.get_ants_data('mni') )
    >>> ants.get_nifti_sform_spatial_info(metadata)
    """
    A = _as_affine_4x4_from_srow(metadata)
    if A is None:
        warnings.warn("No sform present in metadata (srow_x/y/z).", RuntimeWarning)
        return None

    # Save original shear
    spacing, D0 = _spacing_and_dirs_from_affine(A[:3, :3])
    shear = _shear_components_from_dirs(D0)

    try:
        A2 = deshear_nifti_sform(metadata, deshear_threshold=deshear_threshold,
                            max_angle_deviation=max_angle_deviation)
        desheared = not np.allclose(A, A2)
        if desheared:
            A = A2
    except ValueError as e:
        warnings.warn(f"Deshearing sform failed: {e}", RuntimeWarning)
        return None

    # Convert from NIFTI RAS to ITK LPS
    A = _ras_to_lps_affine(A)

    spacing, D = _spacing_and_dirs_from_affine(A[:3, :3])
    origin = A[:3, 3].tolist()
    return dict(
        pixdim_spacing=_pixdims_from_metadata(metadata),
        transform_spacing=spacing,
        origin=origin,
        direction=D.tolist(),
        desheared=desheared,
        original_shear=shear,
    )


def get_nifti_qform_spatial_info(metadata):
    """
    Extract qform-derived spacing, origin, direction. Uses the 4x4 'qto_xyz' from the metadata dict. This is the rotation
    matrix derived from quaternion parameters in the NIfTI header, multiplied by the pixdim scales and with the qoffset
    translation.

    Note: output is in ITK LPS coordinates

    Returns
    -------
    dict with keys:
      pixdim_spacing = [sx, sy, sz]  (from pixdim[1..3])
      transform_spacing : [sx, sy, sz]
      origin  : [ox, oy, oz]
      direction : 3x3 list (direction cosines)

    Example
    -------
    >>> import ants
    >>> metadata = ants.read_image_metadata( ants.get_ants_data('mni') )
    >>> ants.get_nifti_qform_spatial_info(metadata)
    """
    A = _as_affine_4x4_from_qto(metadata)
    if A is None:
        raise ValueError("No qform 'qto_xyz' present in metadata.")

    # Convert from NIFTI RAS to ITK LPS
    A = _ras_to_lps_affine(A)

    spacing, D = _spacing_and_dirs_from_affine(A[:3, :3])
    origin = A[:3, 3].tolist()
    return dict(
        pixdim_spacing=_pixdims_from_metadata(metadata),
        transform_spacing=spacing,
        origin=origin,
        direction=D.tolist())


def get_nifti_spatial_transform_from_metadata(metadata, prefer_sform=True, deshear_threshold=1e-6, max_angle_deviation=0.5,
                                              verbose = False):
    """
    Return a dict containing origin/spacing/direction in ITK (LPS) coordinates, derived from NIfTI header metadata.

    This function only returns the 3D spatial transform information, including spacing for the first
    three dimensions. It does not modify time spacing or direction for 4D images.

    Note that the spacing is always taken from the pixdim fields in the NIfTI header, as is standard in ITK. The transforms
    are checked for consistency with the pixdim spacing, if they are not the same, it indicates that the transform is something
    other than a reorientation to the native physical space.

    If prefer_sform is True (default), the sform transform is used if it is present (sform_code > 0) and either has shear
    beneath the threshold, or the shear can be removed without exceeding the max_axis_deviation.

    If prefer_sform is False, qform is used if it is present (qform_code > 0), otherwise sform is used if present and usable.

    Arguments
    ---------
    image : ants.ANTsImage
        The image to modify.
    metadata : dict
        The NIfTI header metadata dictionary.
    prefer_sform : bool
        Whether to prefer sform over qform if both are available.
    deshear_threshold : float
        Shear threshold for deshearing sform.
    max_angle_deviation : float
        Maximum angle deviation for deshearing sform.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    {
      "origin": [ox, oy, oz],
      "pixdim_spacing": [sx, sy, sz],         # from the pixdims
      "transform_spacing": [sx, sy, sz],  # from the transform
      "direction": [[...],[...],[...]],
      "transform_source": "sform" | "qform"
    }

    Example
    -------
    >>> import ants
    >>> metadata = ants.read_image_metadata( ants.get_ants_data('mni') )
    >>> ants.get_nifti_spatial_transform_from_metadata(metadata)
    """
    have_sform = all(k in metadata for k in ("srow_x", "srow_y", "srow_z")) and int(metadata.get("sform_code", 0)) > 0
    have_qform = ("qto_xyz" in metadata) and int(metadata.get("qform_code", 0)) > 0

    if not have_sform and not have_qform:
        warnings.warn("No sform or qform present in metadata; image not modified", RuntimeWarning)
        return

    # grab expected spacings from NIfTI header (pixdim)
    pixdims = _pixdims_from_metadata(metadata)

    # Try sform if requested/available
    info_s = None
    if prefer_sform and have_sform:
        try:
            info_s = get_nifti_sform_spatial_info(
                metadata,
                deshear_threshold=deshear_threshold,
                max_angle_deviation=max_angle_deviation,
            )
            if verbose:
                print(f"[sform] spacing={info_s['spacing']} origin={info_s['origin']} (desheared={info_s['desheared']})")
            # Verify spacing vs pixdim
            if not _spacing_matches(info_s['transform_spacing'], pixdims):
                warnings.warn(
                    f"sform-derived spacing {info_s['transform_spacing']} does not match NIfTI pixdim {pixdims}; "
                    "ignoring sform and trying qform.",
                    RuntimeWarning,
                )
                info_s = None
        except Exception as e:
            if verbose:
                print(f"[sform] failed: {e}")
            info_s = None

    # If sform not used, try qform
    if info_s is None:
        if not have_qform:
            warnings.warn("No usable sform or qform present in metadata; image not modified", RuntimeWarning)
            return
        info_q = get_nifti_qform_spatial_info(metadata)
        if not _spacing_matches(info_q['transform_spacing'], pixdims):
                raise ValueError(f"qform-derived spacing {info_q['transform_spacing']} does not match NIfTI pixdim {pixdims}")
        if verbose:
            print(f"[qform] spacing={info_q['transform_spacing']} origin={info_q['origin']}")
            print(f"Setting spacing to pixdims={pixdims}")
        return dict(
            origin=info_q['origin'],
            spacing=pixdims,
            direction=info_q['direction'],
            transform_source="qform",
        )
    else:
        # Use sform
        return dict(
            origin=info_s['origin'],
            spacing=pixdims,
            direction=info_s['direction'],
            transform_source="sform",
        )


def set_nifti_spatial_transform_from_metadata(image, metadata, prefer_sform=True, deshear_threshold=1e-6,
                                              max_angle_deviation=0.5, verbose=False):
    """
    Set the spatial transform of an ANTsImage from NIfTI header metadata. This sets the 3D spatial transform but does
    not modify spacing or the fourth row / column of the direction matrix for 4D images. This function does not support 2D
    images, because the projection of a 3D spatial transform to 2D is not preserved in the ANTsImage.

    The spacing is not modified but it is checked for consistency with the pixdim fields in the NIfTI header,
    as is standard in ITK. If the spacing does not match the pixdim, an error is raised.

    Arguments
    ---------
    image : ants.ANTsImage
        The image to modify.
    metadata : dict
        The NIfTI header metadata dictionary.
    prefer_sform : bool
        Whether to prefer sform over qform if both are available.
    deshear_threshold : float
        Shear threshold for deshearing sform, shear below this will be ignored.
    max_angle_deviation : float
        Maximum angle deviation for directions after deshearing sform. If the desheared directions deviate from the original
        directions by more than this value (in degrees), deshearing fails, and qform is used if available.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    ants.ANTsImage
        The modified image with updated spatial transform.

    Example
    -------
    >>> import ants
    >>> img = ants.image_read( ants.get_ants_data('mni') )
    >>> metadata = ants.read_image_metadata( ants.get_ants_data('mni') )
    >>> ants.set_nifti_spatial_transform_from_metadata(img, metadata)
    """
    if image.dimension == 2:
        raise ValueError("Projection of NIFTI spatial orientation to 2D is not supported")

    info = get_nifti_spatial_transform_from_metadata(
        metadata,
        prefer_sform=prefer_sform,
        deshear_threshold=deshear_threshold,
        max_angle_deviation=max_angle_deviation,
        verbose=verbose,
    )

    if info is None:
        raise ValueError("No usable spatial transform found in metadata; image not modified.")

    if image.dimension == 4:
        # Use original definition of time spacing and origin
        origin_4d = info['origin'].copy()
        origin_4d.append(image.origin[-1])
        image.set_origin(origin_4d)
        # direction is 4x4 with identity time
        dir4 = np.eye(4)
        dir4[:3, :3] = np.array(info['direction'])
        image.set_direction(dir4.tolist())
    else:
        image.set_origin(info['origin'])
        image.set_direction(info['direction'])
