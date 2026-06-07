"""Centralized Gemini API helpers for the team_8 pipeline.

This module owns all direct google-genai usage:

1. Natural-language manipulation command -> structured task plan
2. RGB image + object prompt -> normalized Gemini bounding box

ROS2 nodes should import GeminiAPI instead of constructing genai.Client directly.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

try:  # pragma: no cover - runtime dependency
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - import-only test fallback
    genai = None
    types = None


DESTINATIONS = (
    "storage_1",
    "storage_2",
    "bookshelf_floor1",
    "bookshelf_floor2",
    "unspecified",
)

TASK_PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "tasks": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "object": {
                        "type": "STRING",
                        "description": (
                            "Normalized object name in lowercase snake_case, "
                            "for example meat_can or coke_can."
                        ),
                    },
                    "destination": {
                        "type": "STRING",
                        "enum": list(DESTINATIONS),
                        "description": "Canonical destination used by team_8.",
                    },
                },
                "required": ["object", "destination"],
            },
        }
    },
    "required": ["tasks"],
}

DETECTION_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "box_2d": {
                "type": "ARRAY",
                "items": {"type": "INTEGER"},
                "description": (
                    "Bounding box [ymin, xmin, ymax, xmax] scaled from 0 to 1000."
                ),
            },
            "label": {
                "type": "STRING",
                "description": "Descriptive label of the detected item.",
            },
        },
        "required": ["box_2d", "label"],
    },
}


class GeminiAPIError(RuntimeError):
    """Raised when Gemini generation or response validation fails."""


class GeminiAPI:
    """Small reusable wrapper around the Google GenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        logger: Any | None = None,
        max_retries: int = 2,
        retry_delay_sec: float = 1.0,
    ) -> None:
        if genai is None or types is None:
            raise ImportError(
                "google-genai is required. Install requirements/planner-runtime.txt."
            )

        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        self._client = genai.Client(api_key=resolved_key)
        self._model = str(model)
        self._logger = logger
        self._max_retries = max(1, int(max_retries))
        self._retry_delay_sec = max(0.0, float(retry_delay_sec))

    @property
    def model(self) -> str:
        return self._model

    def parse_task_command(self, instruction: str) -> dict[str, Any]:
        """Convert one natural-language command into one task per object.

        The returned dictionary has this form:

        {
            "tasks": [
                {"object": "banana", "destination": "storage_1"},
                ...
            ]
        }

        Destination mapping for the current test_run branch:
          - left storage / left basket  -> storage_1
          - right storage / right basket -> storage_2
          - lower / first shelf -> bookshelf_floor1
          - upper / second shelf -> bookshelf_floor2
          - no destination -> unspecified
        """
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("Task instruction is empty.")

        prompt = f"""
You are the command parser for a ROS2 robotic manipulation pipeline.

Convert the instruction into strict JSON containing exactly one task for each
object mentioned. Preserve object order. If multiple objects share a
destination, still create a separate task for each object.

Normalize object names to lowercase snake_case:
- "meat can" -> "meat_can"
- "coke can" -> "coke_can"

Use only these destination values:
- "storage_1": left storage, left basket
- "storage_2": right storage, right basket
- "bookshelf_floor1": first shelf, lower shelf, shelf floor 1
- "bookshelf_floor2": second shelf, upper shelf, shelf floor 2
- "unspecified": no destination is stated

When the instruction only names an object, use "unspecified".
When it says only "shelf" without a floor, use "bookshelf_floor1".
Do not invent objects.

Instruction:
{instruction}
""".strip()

        payload = self._generate_json(
            contents=prompt,
            schema=TASK_PLAN_SCHEMA,
            temperature=0.0,
        )
        return self._validate_task_plan(payload)

    def localize_object(self, pil_image: Any, prompt: str) -> dict[str, Any]:
        """Locate the first matching object in a PIL image.

        Returns:
            {
                "label": str,
                "box_2d": [ymin, xmin, ymax, xmax],  # normalized 0..1000
                "image_size": [width, height]
            }
        """
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("Object localization prompt is empty.")
        if not hasattr(pil_image, "size"):
            raise TypeError("localize_object expects a PIL image with a size attribute.")

        width, height = pil_image.size
        full_prompt = (
            f"Find the object matching this prompt: {prompt}\n"
            "Return a JSON list. Return the best matching object first. "
            "For each object, return 'label' and 'box_2d'. "
            "'box_2d' must contain exactly four integers in this order: "
            "[ymin, xmin, ymax, xmax], normalized from 0 to 1000."
        )

        payload = self._generate_json(
            contents=[full_prompt, pil_image],
            schema=DETECTION_SCHEMA,
            temperature=0.1,
        )

        if not isinstance(payload, list) or not payload:
            raise GeminiAPIError("Gemini returned no object detections.")

        detection = payload[0]
        if not isinstance(detection, dict):
            raise GeminiAPIError(
                f"Gemini detection must be an object, got {type(detection).__name__}."
            )

        label = str(detection.get("label", "")).strip() or "target"
        raw_box = detection.get("box_2d")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise GeminiAPIError(f"Invalid Gemini box_2d: {raw_box!r}")

        try:
            box_2d = [int(value) for value in raw_box]
        except (TypeError, ValueError) as exc:
            raise GeminiAPIError(f"Non-integer Gemini box_2d: {raw_box!r}") from exc

        box_2d = [max(0, min(1000, value)) for value in box_2d]
        ymin, xmin, ymax, xmax = box_2d
        if ymax <= ymin or xmax <= xmin:
            raise GeminiAPIError(f"Degenerate Gemini box_2d: {box_2d!r}")

        return {
            "label": label,
            "box_2d": box_2d,
            "image_size": [int(width), int(height)],
        }

    def _generate_json(
        self,
        *,
        contents: Any,
        schema: dict[str, Any],
        temperature: float,
    ) -> Any:
        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": float(temperature),
        }

        # test_run currently uses gemini-2.5-flash and thinking_budget=0.
        # Keep this optional so the wrapper remains tolerant of SDK differences.
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0
            )
        except Exception:
            pass

        config = types.GenerateContentConfig(**config_kwargs)
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                text = getattr(response, "text", None)
                if not text:
                    raise GeminiAPIError("Gemini returned an empty response.")
                return json.loads(text)
            except Exception as exc:
                last_error = exc
                self._log(
                    "warning",
                    f"Gemini request failed "
                    f"(attempt {attempt}/{self._max_retries}): {exc}",
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay_sec * attempt)

        raise GeminiAPIError(
            f"Gemini request failed after {self._max_retries} attempt(s): "
            f"{last_error}"
        ) from last_error

    def _validate_task_plan(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise GeminiAPIError(
                f"Task plan must be a JSON object, got {type(payload).__name__}."
            )

        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise GeminiAPIError("Gemini task plan contains no tasks.")

        tasks: list[dict[str, str]] = []
        for index, raw_task in enumerate(raw_tasks):
            if not isinstance(raw_task, dict):
                raise GeminiAPIError(f"Task {index} is not a JSON object.")

            object_name = _normalize_object_name(raw_task.get("object", ""))
            destination = str(raw_task.get("destination", "")).strip()

            if not object_name:
                raise GeminiAPIError(f"Task {index} has an empty object name.")
            if destination not in DESTINATIONS:
                raise GeminiAPIError(
                    f"Task {index} has invalid destination {destination!r}."
                )

            tasks.append(
                {
                    "object": object_name,
                    "destination": destination,
                }
            )

        return {"tasks": tasks}

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        callback = getattr(self._logger, level, None)
        if callable(callback):
            callback(message)


def _normalize_object_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")
