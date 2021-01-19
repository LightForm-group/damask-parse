"""Functions for quaternion manipulation."""

import numpy as np


def euler2quat(euler_angles, degrees=False, P=1):
    """Convert Bunge-convention Euler angles to unit quaternions.

    Parameters
    ----------
    euler_angles : ndarray of shape (N, 3) of float
        Array of N row three-vectors of Euler angles, specified as proper Euler angles in
        the Bunge convention (rotations are about Z, new X, new new Z).
    degrees : bool, optional
        If True, `euler_angles` are expected in degrees, rather than radians.
    P : int, optional
        The "P" constant, either +1 or -1, as defined within [1].

    Returns
    -------
    quats : ndarray of shape (N, 4) of float
        Array of N row four-vectors of unit quaternions.

    Notes
    -----
    Conversion of Bunge Euler angles to quaternions due to Ref. [1].

    References
    ----------
    [1] Rowenhorst, D, A D Rollett, G S Rohrer, M Groeber, M Jackson,
        P J Konijnenberg, and M De Graef. "Consistent Representations
        of and Conversions between 3D Rotations". Modelling and Simulation
        in Materials Science and Engineering 23, no. 8 (1 December 2015):
        083501. https://doi.org/10.1088/0965-0393/23/8/083501.

    """

    if P not in [-1, 1]:
        raise ValueError('P must be -1 or +1')

    if degrees:
        euler_angles = np.deg2rad(euler_angles)

    phi_1 = euler_angles[:, 0]
    Phi = euler_angles[:, 1]
    phi_2 = euler_angles[:, 2]

    sigma = 0.5 * (phi_1 + phi_2)
    delta = 0.5 * (phi_1 - phi_2)
    c = np.cos(Phi / 2)
    s = np.sin(Phi / 2)

    quats = np.array([
        +c * np.cos(sigma),
        -P * s * np.cos(delta),
        -P * s * np.sin(delta),
        -P * c * np.sin(sigma),
    ]).T

    # Move to northern hemisphere:
    quats[quats[:, 0] < 0] *= -1

    return quats


def axang2quat(axis, angle):
    """Convert an axis-angle to a quaternion.

    Parameters
    ----------
    axis : ndarray of shape (3,) of float
        Axis of rotation.
    angle : float
        Angle of rotation in radians.

    Returns
    -------
    quat : ndarray of shape (4,) of float

    Notes
    -----
    Conversion of axis-angle to quaternion due to Ref. [1].

    References
    ----------
    [1] Rowenhorst, D, A D Rollett, G S Rohrer, M Groeber, M Jackson,
        P J Konijnenberg, and M De Graef. "Consistent Representations
        of and Conversions between 3D Rotations". Modelling and Simulation
        in Materials Science and Engineering 23, no. 8 (1 December 2015):
        083501. https://doi.org/10.1088/0965-0393/23/8/083501.            

    """

    axis = axis / np.linalg.norm(axis)
    quat = np.zeros(4)
    quat[0] = np.cos(angle / 2)
    quat[1:] = np.sin(angle / 2) * axis

    return quat


def multiply_quaternions(q1, q2, P=1):
    """Find the product of two quaternions.

    Parameters
    ----------
    q1 : ndarray of shape (4,)
    q2 : ndarray of shape (4,)
    P : int, optional
        The "P" constant, either +1 or -1, as defined within [1].    

    Returns
    -------
    q3 : ndarray of shape (4,)

    References
    ----------
    [1] Rowenhorst, D, A D Rollett, G S Rohrer, M Groeber, M Jackson,
        P J Konijnenberg, and M De Graef. "Consistent Representations
        of and Conversions between 3D Rotations". Modelling and Simulation
        in Materials Science and Engineering 23, no. 8 (1 December 2015):
        083501. https://doi.org/10.1088/0965-0393/23/8/083501.  

    """

    s1, v1 = q1[0], q1[1:]
    s2, v2 = q2[0], q2[1:]

    q3 = np.zeros(4)
    q3[0] = (s1 * s2) - np.dot(v1, v2)
    q3[1:] = (s1 * v2) + (s2 * v1) + P * np.cross(v1, v2)

    return q3
