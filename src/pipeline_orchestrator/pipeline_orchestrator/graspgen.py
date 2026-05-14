from sensor_msgs.msg import Image


class GraspGen:
    """GraspGen grasp pose generation module."""

    def __init__(self, logger):
        self._logger = logger
        # TODO: load GraspGen diffusion model

    def generate_grasp(self, masks, depth: Image):
        """Generate grasp pose from segmentation masks and depth image. Returns pose or None."""
        # TODO: run GraspGen diffusion model on masks + depth
        self._logger.warn('GraspGen.generate_grasp not yet implemented.')
        return None
