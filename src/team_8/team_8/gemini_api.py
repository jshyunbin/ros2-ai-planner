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
                    },
                },
                "required": ["object", "destination"],
            },
        }
    },
    "required": ["tasks"],
}

TASK_VERIFICATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "present_in_source_workspace": {
            "type": "BOOLEAN",
            "description": (
                "True only when the requested object is still visible in the "
                "original pickup workspace/table area."
            ),
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence from 0.0 to 1.0.",
        },
        "reason": {
            "type": "STRING",
            "description": "Brief visual reason for the decision.",
        },
    },
    "required": [
        "present_in_source_workspace",
        "confidence",
        "reason",
    ],
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
   - bookshelf_floor1: first shelf, lower shelf, or unspecified shelf
   - bookshelf_floor2: second shelf or upper shelf
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

    def verify_object_removed(
        self,
        pil_image: Any,
        *,
        object_name: str,
        destination: str,
    ) -> dict[str, Any]:
        """Check whether an object remains in the original pickup workspace.

        The image must be captured after the robot has returned home. The
        destination area is explicitly excluded, so an object correctly placed
        in a basket or bookshelf is not treated as a failed pick.
        """
        if not hasattr(pil_image, "size"):
            raise TypeError(
                "verify_object_removed expects a PIL image with a size attribute."
            )

        normalized_object = _normalize_object_name(object_name)
        if not normalized_object:
            raise ValueError("Verification object name is empty.")

        display_name = normalized_object.replace("_", " ")
        destination = str(destination).strip() or "unspecified"

        prompt = f"""
You are verifying the result of a robotic pick-and-place task.

Target object: {display_name}
Intended destination: {destination}

The image was captured after the robot returned to its home pose.
Decide whether the target object is STILL PRESENT IN THE ORIGINAL PICKUP
WORKSPACE, meaning the main table/work area where loose objects are picked up.

Important rules:
1. Ignore the robot arm and gripper.
2. Ignore the target object if it is visible inside the intended destination
   basket, storage area, or bookshelf. A correctly placed object at the
   destination means present_in_source_workspace must be false.
3. Do not report another similar object unless it clearly matches the target.
4. If the target is still lying in the pickup workspace, return true.
5. If the target is absent from the pickup workspace, return false.
6. If visibility is ambiguous, return true so the robot can retry safely.

Return strict JSON with:
- present_in_source_workspace: boolean
- confidence: number from 0.0 to 1.0
- reason: one short sentence
""".strip()

        payload = self._generate_json(
            contents=[prompt, pil_image],
            schema=TASK_VERIFICATION_SCHEMA,
            temperature=0.0,
        )

        if not isinstance(payload, dict):
            raise GeminiAPIError(
                "Task verification response must be a JSON object."
            )

        present = payload.get("present_in_source_workspace")
        if not isinstance(present, bool):
            raise GeminiAPIError(
                "Verification field present_in_source_workspace must be boolean."
            )

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError) as exc:
            raise GeminiAPIError(
                f"Invalid verification confidence: {payload.get('confidence')!r}"
            ) from exc

        confidence = max(0.0, min(1.0, confidence))
        reason = str(payload.get("reason", "")).strip()

        return {
            "present_in_source_workspace": present,
            "confidence": confidence,
            "reason": reason,
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