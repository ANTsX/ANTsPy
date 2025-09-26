__all__ = [
    "deshear_sform",
    "get_sform_shear",
    "get_sform_image_info",
    "get_qform_image_info",
    "get_transform_from_metadata",
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


def _closest_orthogonal_dirs(M, enforce_right_handed=True):
    """
    Return the orthogonal matrix Q that is closest (Frobenius) to M.
    Uses SVD-based polar decomposition: M = U S V^T, Q = U V^T.
    If enforce_right_handed, make det(Q)=+1 by flipping the last singular vector.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError("closest_orthogonal_dirs expects a 3x3 matrix.")

    # SVD
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Q = U @ Vt

    # Optionally enforce a proper rotation (no reflection)
    if enforce_right_handed and np.linalg.det(Q) < 0:
        # Flip the last column of U (equivalently, last row of Vt)
        U[:, -1] *= -1
        Q = U @ Vt

    return Q


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
    """
    A = _as_affine_4x4_from_srow(metadata)
    if A is None:
        raise ValueError("No sform found (srow_x/y/z missing)")
    _, D = _spacing_and_dirs_from_affine(A[:3, :3])
    return _shear_components_from_dirs(D)


def deshear_nifti_sform(metadata, shear_threshold=1e-5, max_axis_deviation=1.0):
    """
    Deshear an sform matrix in the provided metadata dict.
    Returns a new 4x4 affine (index->RAS) with shear removed (direction orthonormal),
    preserving per-axis spacings and translation. Raises ValueError if the orthonormalized
    axes deviate from the originals by more than max_axis_deviation degrees.
    """
    A = _as_affine_4x4_from_srow(metadata)
    if A is None:
        raise ValueError("No sform found (srow_x/y/z missing)")

    spacing, D = _spacing_and_dirs_from_affine(A[:3, :3])
    shear_xy, shear_xz, shear_yz = _shear_components_from_dirs(D)
    shear_max = max(abs(shear_xy), abs(shear_xz), abs(shear_yz))

    if shear_max <= shear_threshold:
        # Already fine
        return A.copy()

    Q = _closest_orthogonal_dirs(D)

    # sanity check: how much do directions deviate?
    dev = max(
        _angle_deg(D[:, 0], Q[:, 0]),
        _angle_deg(D[:, 1], Q[:, 1]),
        _angle_deg(D[:, 2], Q[:, 2]),
    )
    if dev > max_axis_deviation:
        raise ValueError(
            f"Desheared axes deviate by {dev:.3f} deg (> max allowed {max_axis_deviation} deg). "
            "This likely indicates significant obliquity/shear"
        )

    # Rebuild affine with same translation
    S = np.diag(spacing)
    A_new = np.eye(4, dtype=float)
    A_new[:3, :3] = Q @ S
    A_new[:3, 3]  = A[:3, 3]
    return A_new


def get_nifti_sform_spatial_info(metadata, shear_threshold=1e-6, max_axis_deviation=1.0):
    """
    Extract sform-derived spacing, origin, direction.

    Output is in ITK LPS coordinates.

    Returns
    -------
    dict with keys:
      spacing : [sx, sy, sz]
      origin  : [ox, oy, oz]
      direction : 3x3 list (direction cosines)
      desheared : bool
      original_shear : [shear_xy, shear_xz, shear_yz]
    """
    A = _as_affine_4x4_from_srow(metadata)
    if A is None:
        warnings.warn("No sform present in metadata (srow_x/y/z).", RuntimeWarning)
        return None

    # Save original shear
    spacing, D0 = _spacing_and_dirs_from_affine(A[:3, :3])
    shear = _shear_components_from_dirs(D0)

    try:
        A2 = deshear_nifti_sform(metadata, shear_threshold=shear_threshold,
                            max_axis_deviation=max_axis_deviation)
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
        spacing=spacing,
        origin=origin,
        direction=D.tolist(),
        desheared=desheared,
        original_shear=shear,
    )


def get_nifti_qform_spatial_info(metadata):
    """
    Extract qform-derived spacing, origin, direction. Uses 'qto_xyz' from the metadata dict.

    Note: output is in ITK LPS coordinates

    Returns
    -------
    dict with keys:
      spacing : [sx, sy, sz]
      origin  : [ox, oy, oz]
      direction : 3x3 list (direction cosines)
    """
    A = _as_affine_4x4_from_qto(metadata)
    if A is None:
        raise ValueError("No qform 'qto_xyz' present in metadata.")

    # Convert from NIFTI RAS to ITK LPS
    A = _ras_to_lps_affine(A)

    spacing, D = _spacing_and_dirs_from_affine(A[:3, :3])
    origin = A[:3, 3].tolist()
    return dict(spacing=spacing, origin=origin, direction=D.tolist())


def get_nifti_spatial_transform_from_metadata(metadata: dict, prefer_sform: bool = True, shear_threshold: float = 1e-6,
                                              max_axis_deviation: float = 1.0, verbose: bool = False):
    """
    Return a dict containing origin/spacing/direction (LPS) derived from NIfTI header metadata.

    This function only returns the 3D spatial transform information.

    Returns
    -------
    {
      "origin": [ox, oy, oz],
      "spacing": [sx, sy, sz],         # from the chosen transform (not altering image)
      "direction": [[...],[...],[...]],
      "transform_source": "sform" | "qform"
    }
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
                shear_threshold=shear_threshold,
                max_axis_deviation=max_axis_deviation,
            )
            if verbose:
                print(f"[sform] spacing={info_s['spacing']} origin={info_s['origin']} (desheared={info_s['desheared']})")
            # Verify spacing vs pixdim
            if not _spacing_matches(info_s["spacing"], pixdims):
                warnings.warn(
                    f"sform-derived spacing {info_s['spacing']} does not match NIfTI pixdim {pixdims}; "
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
        if not _spacing_matches(info_q["spacing"], pixdims):
                raise ValueError(f"qform-derived spacing {info_q['spacing']} does not match NIfTI pixdim {pixdims}")
        if verbose:
            print(f"[qform] spacing={info_q['spacing']} origin={info_q['origin']}")
        return dict(
            origin=info_q["origin"],
            spacing=info_q["spacing"],
            direction=info_q["direction"],
            transform_source="qform",
        )
    else:
        # Use sform
        return dict(
            origin=info_s["origin"],
            spacing=info_s["spacing"],
            direction=info_s["direction"],
            transform_source="sform",
        )


def set_nifti_spatial_transform_from_metadata(image, metadata: dict, prefer_sform: bool = True, shear_threshold: float = 1e-6,
                                              max_axis_deviation: float = 1.0, verbose: bool = False):
    """
    Set the spatial transform of an ANTsImage from NIfTI header metadata. This sets the 3D spatial transform but does
    not modify time spacing or direction for 4D images.

    Parameters
    ----------
    image : ants.ANTsImage
        The image to modify.
    metadata : dict
        The NIfTI header metadata dictionary.
    prefer_sform : bool
        Whether to prefer sform over qform if both are available.
    shear_threshold : float
        Shear threshold for deshearing sform.
    max_axis_deviation : float
        Maximum axis deviation for deshearing sform.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    ants.ANTsImage
        The modified image with updated spatial transform.
    """
    info = get_nifti_spatial_transform_from_metadata(
        metadata,
        prefer_sform=prefer_sform,
        shear_threshold=shear_threshold,
        max_axis_deviation=max_axis_deviation,
        verbose=verbose,
    )

    if image.dimension == 2:
        raise ValueError("Projection of NIFTI spatial orientation to 2D is not supported")

    # check spacing matches metadata pixdim
    pixdims = _pixdims_from_metadata(metadata)
    if not _spacing_matches(image.spacing[:3], pixdims):
        raise ValueError(f"Image spacing {image.spacing} does not match NIfTI pixdim {pixdims}; cannot set transform.")

    if image.dimension == 4:
        # Use original definition of time spacing and origin
        image.set_origin(info["origin"].append(image.origin[-1]))
        # direction is 4x4 with identity time
        dir4 = np.eye(4)
        dir4[:3, :3] = np.array(info["direction"])
        image.set_direction(dir4.tolist())
    else:
        image.set_origin(info["origin"])
        image.set_direction(info["direction"])
