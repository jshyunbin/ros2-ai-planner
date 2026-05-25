from sensor_msgs.msg import Image


class Sam2:
    """SAM2 segmentation module."""

    def __init__(self, logger):
        self._logger = logger
        # TODO: load SAM2 model (hydra config + checkpoint)

    def segment(
        self,
        rgb: Image,
        prompt: str,
        bbox: tuple[int, int, int, int] | None = None,
    ):
        """Run SAM2 on rgb image.

        If bbox is provided (x1, y1, x2, y2 in pixels), uses it as a spatial
        prompt for higher accuracy. Otherwise falls back to text prompt alone.
        Returns masks or None on failure.
        """
        # TODO: convert sensor_msgs/Image to numpy array
        # TODO: if bbox provided, use predictor.set_image() + predictor.predict(box=bbox)
        # TODO: else use SAM2AutomaticMaskGenerator with text prompt
        self._logger.warn('Sam2.segment not yet implemented.')
        return None
