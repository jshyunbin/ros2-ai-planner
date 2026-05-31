import numpy as np


class GraspGen:
    """GraspGen grasp pose generation module."""

    def __init__(self, logger):
        self._logger = logger
        # TODO: load GraspGen diffusion model

    def generate_grasp(self, point_cloud: np.ndarray):
        """Generate grasp pose from segmented object point cloud.

        Args:
            point_cloud: (N, 3) float32 array of object surface points
                         in robot base frame.

        Returns grasp pose or None on failure.
        """
        # TODO: run GraspGen diffusion model on point_cloud
        self._logger.warn('GraspGen.generate_grasp not yet implemented.')
        return None
