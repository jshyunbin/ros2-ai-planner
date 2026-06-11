"""Gemini command parser for the team_8 manipulation pipeline.

This module only converts natural-language manipulation instructions into a
strict JSON-compatible task plan.

The existing Gemini/SAM2 localization logic in segmentation_service.py remains
independent and unchanged.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


DESTINATIONS = (
    "storage_1",
    "storage_2",
    "bookshelf",
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
                    },
                },
                "required": ["object", "destination"],
            },
        }
    },
    "required": ["tasks"],
}

OBJECT_COUNT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "count": {
            "type": "INTEGER",
            "description": (
                "How many instances of the target object type are visible in "
                "the source pickup workspace. 0 if none remain."
            ),
        },
        "reason": {
            "type": "STRING",
            "description": "Brief visual reason for the count.",
        },
    },
    "required": ["count", "reason"],
}


class GeminiAPIError(RuntimeError):
    """Raised when Gemini generation or validation fails."""


class GeminiAPI:
    """Gemini wrapper used only for natural-language task parsing."""

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
                "google-genai is required. "
                "Install requirements/planner-runtime.txt."
            )

        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        resolved_key = resolved_key.strip()
        if not resolved_key.isascii():
            raise ValueError(
                "GEMINI_API_KEY contains non-ASCII characters. "
                "Set the actual key issued by Google AI Studio."
            )

        self._client = genai.Client(api_key=resolved_key)
        self._model = str(model)
        self._logger = logger
        self._max_retries = max(1, int(max_retries))
        self._retry_delay_sec = max(0.0, float(retry_delay_sec))

    @property
    def model(self) -> str:
        return self._model

    def parse_task_command(self, instruction: str) -> dict[str, Any]:
        """Convert one instruction into one task per mentioned object."""
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("Task instruction is empty.")

        prompt = f"""
You are the command parser for a ROS2 robotic manipulation pipeline.

Convert the instruction into strict JSON containing exactly one task for each
object mentioned.

Rules:
1. Preserve the order in which objects appear.
2. If multiple objects share a destination, create one task per object.
3. Normalize object names to lowercase snake_case.
4. Do not invent objects.
5. Use only these destination values:
   - storage_1: left storage or left basket
   - storage_2: right storage or right basket
   - bookshelf: any shelf of the bookshelf (lower, upper, first, second, or unspecified shelf)
   - unspecified: no destination was stated
6. Examples:
   - meat can -> meat_can
   - coke can -> coke_can
   - Move the banana to the left storage.
     -> banana, storage_1
   - banana
     -> banana, unspecified

Instruction:
{instruction}
""".strip()

        payload = self._generate_json(
            contents=prompt,
            schema=TASK_PLAN_SCHEMA,
            temperature=0.0,
        )
        return self._validate_task_plan(payload)

    def count_objects(
        self,
        pil_image: Any,
        *,
        object_name: str,
    ) -> dict[str, Any]:
        """Count how many instances of object_name remain in the pickup area.

        The image must be captured with the arm at home so the wrist camera sees
        the source workspace. The prompt instructs Gemini to ignore any
        destination area (basket/storage/bookshelf) and all non-target objects,
        so the count reflects only target-type instances still awaiting pickup.
        """
        if not hasattr(pil_image, "size"):
            raise TypeError(
                "count_objects expects a PIL image with a size attribute."
            )

        normalized_object = _normalize_object_name(object_name)
        if not normalized_object:
            raise ValueError("Count object name is empty.")

        display_name = normalized_object.replace("_", " ")

        prompt = f"""
You are counting objects after a robotic pick-and-place task.

Target object type: {display_name}

The image was captured with the robot arm at its home pose, looking down at the
source pickup workspace (the main table/work area where loose objects are
picked up).

Count how many instances of the target object type are STILL PRESENT IN THE
SOURCE PICKUP WORKSPACE.

Important rules:
1. Ignore the robot arm and gripper.
2. Do not count instances that are inside the destination basket, storage area,
   or bookshelf. Count only instances loose in the pickup workspace.
3. Count only the target object type. Ignore every other kind of object.
4. If none are visible, return 0.

Return strict JSON with:
- count: integer number of target instances in the pickup workspace (>= 0)
- reason: one short sentence
""".strip()

        payload = self._generate_json(
            contents=[prompt, pil_image],
            schema=OBJECT_COUNT_SCHEMA,
            temperature=0.0,
        )

        if not isinstance(payload, dict):
            raise GeminiAPIError("Object count response must be a JSON object.")

        raw_count = payload.get("count")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise GeminiAPIError(
                f"Count field must be an integer, got {raw_count!r}."
            )

        count = max(0, raw_count)
        reason = str(payload.get("reason", "")).strip()

        return {"count": count, "reason": reason}

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
            raise GeminiAPIError("Task plan must be a JSON object.")

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
                    f"Task {index} has invalid destination: {destination!r}"
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