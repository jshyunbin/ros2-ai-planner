from sensor_msgs.msg import Image


class GeminiLocalizer:
    """Gemini Vision API — locates target object in overhead RGB image.

    Returns a bounding box (x1, y1, x2, y2) in pixel coordinates.
    Implementation handled by separate teammate.
    """

    def __init__(self, logger):
        self._logger = logger
        # TODO: initialize google-generativeai client with API key

    def locate_object(
        self,
        rgb_image: Image,
        task_prompt: str,
    ) -> tuple[int, int, int, int] | None:
        """Query Gemini Vision with overhead RGB and task prompt.

        Returns (x1, y1, x2, y2) bounding box in pixel coords, or None on failure.
        """
        # TODO: convert sensor_msgs/Image to PIL Image, call Gemini Vision API
        self._logger.warn('GeminiLocalizer.locate_object not yet implemented.')
        return None
