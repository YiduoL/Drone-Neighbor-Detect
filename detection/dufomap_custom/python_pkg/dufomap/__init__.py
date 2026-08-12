'''
Copied verbatim from the official "dufomap" PyPI package's dufomap/__init__.py
(v1.1.1, pure Python, no compiled dependency) -- the only intentional change is that
this loads dufomap_bind from our own custom-built extension (see
detection/dufomap_custom/src/bind.cpp) instead of the official one, which has no
aarch64 wheel. See src/bind.cpp's header comment for the full story.

Original file header:
# @date: 2024-09-26 10:33
# @author: Qingwen Zhang  (https://kin-zhang.github.io/)
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
#
# This file is part of DUFOMap (https://github.com/Kin-Zhang/dufomap) and SeFlow (https://github.com/KTH-RPL/SeFlow).
# If you find this repo helpful, please cite the respective publication as
# listed on the above website.
'''

__version__ = "1.1.1"

import numpy as np
import dufomap_bind
from dufomap.utils.pose import pose_check

class dufomap:
    def __init__(self, resolution, d_s, d_p, num_threads=0, hit_extension=True, ray_passthrough_hits=False):
        """
        Args:
            resolution (double): resolution of the dufomap
            d_s: inflate hits distance
            d_p: inflate unknown

            # optional, no need to change in most cases
            num_threads: 0: auto (max-thread we can use), 1: single thread, >1: multi-threads
            hit_extension (bool): whether to extend the hit points
            ray_passthrough_hits (bool): whether allow ray pass through the hit area
        """
        self.dufomap = dufomap_bind._dufomap(resolution, d_s, d_p, num_threads, hit_extension, ray_passthrough_hits)

    def run(self, points, pose, cloud_transform=True):
        """
        Parameters
        ----------
        points (np.ndarray):
            point cloud data, shape (N, 3), 3 for x, y, z
        pose (np.ndarray):
            pose of the sensor, two formats are supported:
                * shape (4, 4), 4x4 matrix
                * shape (7,), [x, y, z, qw, qx, qy, qz]
        cloud_transform (bool):
            whether need to transform the point cloud to the world frame
            if the point cloud is already in the world frame, set it to False
        """
        points = np.ascontiguousarray(points.astype(np.float32))
        pose = pose_check(pose)
        self.dufomap.run(points, pose, cloud_transform)

    def segment(self, points, pose, cloud_transform=True):
        """
        Parameters
        ----------
        points (np.ndarray):
            point cloud data, shape (N, 3), 3 for x, y, z
        pose (np.ndarray):
            pose of the sensor, two formats are supported:
                * shape (4, 4), 4x4 matrix
                * shape (7,), [x, y, z, qw, qx, qy, qz]
        cloud_transform (bool):
            whether need to transform the point cloud to the world frame
            if the point cloud is already in the world frame, set it to False

        Returns
        -------
        labels (np.ndarray):
            dynamic binary labels, shape (N,); 0 for static, 1 for dynamic
        """
        points = np.ascontiguousarray(points.astype(np.float32))
        pose = pose_check(pose)
        return np.array(self.dufomap.segment(points, pose, cloud_transform)).astype(np.uint8)

    def outputMap(self, points, voxel_map=False, file_name="dufomap_output"):
        points = np.ascontiguousarray(points.astype(np.float32))
        self.dufomap.outputMap(points, voxel_map, file_name)

    def oncePropagateCluster(self, if_propagate=False, if_cluster=False):
        self.dufomap.oncePropagateCluster(if_propagate, if_cluster)

    def setCluster(self, depth, min_points, max_dis):
        self.dufomap.setCluster(depth, min_points, max_dis)

    def printDetailTiming(self):
        self.dufomap.printDetailTiming()

    def clean(self):
        self.dufomap.clean()
