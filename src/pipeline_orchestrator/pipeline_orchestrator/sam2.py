from sensor_msgs.msg import Image


class Sam2:
    """SAM2 segmentation module."""

    def __init__(self, logger):
        self._logger = logger
        # TODO: load SAM2 model (hydra config + checkpoint)

    def segment(self, rgb: Image, prompt: str):
        """Run SAM2 on rgb image with text prompt. Returns masks or None on failure."""
        # TODO: convert sensor_msgs/Image to numpy, run SAM2 inference
        self._logger.warn('Sam2.segment not yet implemented.')
        return None
