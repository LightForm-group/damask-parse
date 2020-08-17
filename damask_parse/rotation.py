"""These functions have been borrowed from other Python packages:
    - vec-maths (https://github.com/aplowman/vec-maths/) and
    - crystex (https://github.com/maria-yankova/crystallographic-texture)
"""

import numpy as np


def ax_ang2rot_mat(axes, angles, degrees=False):
    """
    Generates pre-multiplication rotation matrices for given axes and angles.

    Parameters
    ----------
    axes : ndarray
        Array of shape (N, 3), which if N is 1, will be tiled to the size
        (M, 3). Otherwise, N must be equal to M (for M, see `angles`).
    angles : ndarray
        Array of shape (M).
    degrees : bool (optional)
        If True, `angles` interpreted as degrees.

    Returns
    -------
    ndarray of shape (N or M, 3, 3).

    Notes
    -----
    Computed using the Rodrigues' rotation formula.

    Examples
    --------

    Find the rotation matrix for a single axis and angle:

    >>> ax_ang2rot_mat(np.array([[0,0,1]]), np.array([np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]]])

    Find the rotation matrices for different angles about the same axis:

    >>> ax_ang2rot_mat(np.array([[0,0,1]]), np.array([np.pi/4, -np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]],

           [[ 0.70710678,  0.70710678,  0.        ],
            [-0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]]])

    Find the rotation matrices about different axes by the same angle:

    >>> ax_ang2rot_mat(np.array([[0,0,1], [0,1,0]]), np.array([np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]],

           [[ 0.70710678,  0.        ,  0.70710678],
            [ 0.        ,  1.        ,  0.        ],
            [-0.70710678,  0.        ,  0.70710678]]])

    Find the rotation matrices about different axes and angles:

    >>> ax_ang2rot_mat(
        np.array([[0,0,1], [0,1,0]]), np.array([np.pi/4, -np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]],

           [[ 0.70710678,  0.        , -0.70710678],
            [ 0.        ,  1.        ,  0.        ],
            [ 0.70710678,  0.        ,  0.70710678]]])

    """

    # Check dimensions

    if axes.ndim == 1:
        axes = axes[np.newaxis]

    angles_err_msg = '`angles` must be a number or array of shape (M,).'

    if isinstance(angles, np.ndarray):
        if angles.ndim != 1:
            raise ValueError(angles_err_msg)

    else:
        try:
            angles = np.array([angles])

        except ValueError:
            print(angles_err_msg)

    if axes.shape[0] == angles.shape[0]:
        n = axes.shape[0]
    else:
        if axes.shape[0] == 1:
            n = angles.shape[0]
        elif angles.shape[0] == 1:
            n = axes.shape[0]
        else:
            raise ValueError(
                'Incompatible dimensions: the first dimension of `axes` or'
                '`angles` must be one otherwise the first dimensions of `axes`'
                'and `angles` must be equal.')

    # Convert to radians if necessary
    if degrees:
        angles = np.deg2rad(angles)

    # Normalise axes to unit vectors:
    axes = axes / np.linalg.norm(axes, axis=1)[:, np.newaxis]

    cross_prod_mat = np.zeros((n, 3, 3))
    cross_prod_mat[:, 0, 1] = -axes[:, 2]
    cross_prod_mat[:, 0, 2] = axes[:, 1]
    cross_prod_mat[:, 1, 0] = axes[:, 2]
    cross_prod_mat[:, 1, 2] = -axes[:, 0]
    cross_prod_mat[:, 2, 0] = -axes[:, 1]
    cross_prod_mat[:, 2, 1] = axes[:, 0]

    rot_mats = np.tile(np.eye(3), (n, 1, 1)) + (
        np.sin(angles)[:, np.newaxis, np.newaxis] * cross_prod_mat) + (
            (1 - np.cos(angles)[:, np.newaxis, np.newaxis]) * (
                cross_prod_mat @ cross_prod_mat))

    return rot_mats


def euler2rot_mat_n(angles, degrees=False):
    """
    Converts sets of Euler angles in Bunge convention to rotation matrices.

    Parameters
    ----------
    angles : ndarray
        An array of shape (N, 3) of Euler angles using Bunge (zx'z")
        convention (φ1, Φ, φ2).
    degrees : bool
        Specify whether units of angles are radians (default) or degrees.

    Returns
    -------
    ndarray
        An array of shape (N, 3, 3).

    Notes
    -----
    Angular ranges are φ1: [0, 2π], Φ: [0, π], φ2: [0, 2π].
    By definition the Euler angles in Bunge convention represent a rotation of
    the reference frame, i.e. a passive transformation. ax_ang2rot_mat() is
    constructed for an active rotations and therefore we apply the rotations
    of opposite sign and in the opposite order (-φ2, -Φ, -φ1) [1].

    [1] Rowenhorst et al. (2015) 23(8), 83501.
        doi.org/10.1088/0965-0393/23/8/083501

    """

    # Add dimension for 1D
    if angles.ndim == 1:
        angles = angles[np.newaxis]

    # Degrees option
    if degrees:
        angles = np.radians(angles)

    #  Euler angles:
    φ1 = angles[:, 0]
    Φ = angles[:, 1]
    φ2 = angles[:, 2]

    Rz_phi1 = ax_ang2rot_mat(np.array([[0, 0, 1]]), -angles[:, 0])
    Rx_Phi = ax_ang2rot_mat(np.array([[1, 0, 0]]), -angles[:, 1])
    Rz_phi2 = ax_ang2rot_mat(np.array([[0, 0, 1]]), -angles[:, 2])

    return Rz_phi2 @ Rx_Phi @ Rz_phi1


def rot_mat2euler(R):
    """
    Converts a rotation matrix to a set of three Euler angles using Bunge
    (zx'z") convention (φ1, Φ, φ2).

    Parameters
    ----------
    R : ndarray
        An array of size (3,3) representing a rotation matrix.

    Returns
    -------
    ndarray
        An array of size (1,3) with the Euler angles according to Bunge
        convention (φ1, Φ, φ2).

    Notes
    -----
    Angular ranges are φ1: [0, 2π], Φ: [0, π], φ2: [0, 2π]
    Not vectorised yet - only works for a single rotation matrix.

    If cos φ1 = c1, cos Φ = c2, cos φ2 = c3,
    sin φ1 = s1, sin Φ = s2, sin φ2 = s3
    The angles are derived from the rotation matrix:
    R = [[ c1c3 - s1c2s3,   s1c3 + c1c2s3,   s2s3 ]
         [ - c1s3 - s1c2c3, - s1s3 + c1c2c3, s2c3 ]
         [ s1s2,            - c1s2,          c2   ]]

    """

    φ1 = np.arctan2(R[2, 0], -R[2, 1])
    Φ = np.arccos(R[2, 2])
    φ2 = np.arctan2(R[0, 2], R[1, 2])

    return np.array([φ1, Φ, φ2])
