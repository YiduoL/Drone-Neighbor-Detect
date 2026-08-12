# Copied verbatim from the official "dufomap" PyPI package's dufomap/utils/pose.py
# (v1.1.1, pure Python/numpy, no compiled or platform-specific dependency). See
# detection/dufomap_custom/src/bind.cpp's header comment for why this local copy
# exists.

from dufomap.utils import SE3
import numpy as np

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Args:
    R : numpy.ndarray
        3x3 rotation matrix

    Returns:
    numpy.ndarray
        Quaternion in format [x, y, z, w]
    """
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qx, qy, qz, qw])

def transform_to_array(pose):
    pose_se3 = SE3(rotation=pose[:3,:3], translation=pose[:3,3])
    qxyzw =rotation_matrix_to_quaternion(pose_se3.rotation)
    pose_array = [pose_se3.translation[0], pose_se3.translation[1], pose_se3.translation[2], \
        qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]
    return pose_array

def pose_check(pose):
    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        return transform_to_array(pose)
    elif isinstance(pose, list) and len(pose) == 7:
        return pose # since it's correct format we want.
    else:
        raise ValueError("Invalid pose format. Expected 4x4 numpy array or list of length 7.")
