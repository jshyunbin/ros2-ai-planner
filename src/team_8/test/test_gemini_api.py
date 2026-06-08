from unittest.mock import MagicMock

import pytest


def _bare_api():
    """A GeminiAPI without __init__ (no real client/key needed)."""
    from team_8.gemini_api import GeminiAPI
    api = GeminiAPI.__new__(GeminiAPI)
    api._logger = None
    return api


def test_count_objects_returns_count_and_reason():
    api = _bare_api()
    api._generate_json = MagicMock(
        return_value={"count": 2, "reason": "two coke cans on the table"})
    image = MagicMock()
    image.size = (640, 480)

    result = api.count_objects(image, object_name="coke_can")

    assert result == {"count": 2, "reason": "two coke cans on the table"}
    # The image and a prompt mentioning the display name are sent to Gemini.
    contents = api._generate_json.call_args.kwargs["contents"]
    assert image in contents
    assert any("coke can" in str(c) for c in contents)


def test_count_objects_clamps_negative_to_zero():
    api = _bare_api()
    api._generate_json = MagicMock(return_value={"count": -3, "reason": "x"})
    image = MagicMock(); image.size = (1, 1)

    assert api.count_objects(image, object_name="banana")["count"] == 0


def test_count_objects_rejects_non_integer_count():
    from team_8.gemini_api import GeminiAPIError
    api = _bare_api()
    api._generate_json = MagicMock(return_value={"count": "lots", "reason": "x"})
    image = MagicMock(); image.size = (1, 1)

    with pytest.raises(GeminiAPIError):
        api.count_objects(image, object_name="banana")


def test_count_objects_rejects_boolean_count():
    from team_8.gemini_api import GeminiAPIError
    api = _bare_api()
    api._generate_json = MagicMock(return_value={"count": True, "reason": "x"})
    image = MagicMock(); image.size = (1, 1)

    with pytest.raises(GeminiAPIError):
        api.count_objects(image, object_name="banana")


def test_count_objects_requires_pil_image():
    api = _bare_api()
    with pytest.raises(TypeError):
        api.count_objects(object(), object_name="banana")


def test_count_objects_rejects_empty_object_name():
    api = _bare_api()
    api._generate_json = MagicMock()
    image = MagicMock(); image.size = (1, 1)

    with pytest.raises(ValueError):
        api.count_objects(image, object_name="   ")
    api._generate_json.assert_not_called()
