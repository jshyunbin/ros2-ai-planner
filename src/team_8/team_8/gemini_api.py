"""Gemini command parser for the team_8 manipulation pipeline.

This module converts natural-language manipulation instructions into a strict
JSON-compatible task plan (parse_task_command) and verifies post-pick success
(verify_object_removed).

The Gemini/SAM2 localization logic in segmentation_service.py remains
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


# ── Object descriptions ───────────────────────────────────────────────────────
# Used to disambiguate visually similar objects in *all* Gemini prompts:
# task parsing, segmentation, and post-pick verification.

OBJECT_DESCRIPTIONS: dict[str, str] = {
    "meat_can": (
        "A rectangular, block-shaped meat can. "
        "It is a wide cuboid with flat sides and slightly rounded corners. "
        "Look for a blue/yellow label or a flat metallic top."
    ),
    "coke_can": (
        "A solid red cylindrical can lying on its side. "
        "Look for a distinct circular red face."
    ),
}

# Single disambiguation paragraph injected into every prompt that could
# confuse similar-looking objects.
OBJECT_DISAMBIGUATION_NOTE = (
    "Use the following object descriptions to tell similar-looking objects apart. "
    "Be careful NOT to confuse the meat can with the coke can:\n"
    "- meat can: A rectangular, block-shaped meat can. It is a wide cuboid with "
    "flat sides and slightly rounded corners. Look for a blue/yellow label or a "
    "flat metallic top.\n"
    "- coke can: A solid red cylindrical can lying on its side. "
    "Look for a distinct circular red face."
)


# ── Destination constants ─────────────────────────────────────────────────────

DESTINATIONS = (
    "storage_1",
    "storage_2",
    "bookshelf_floor1",
    "bookshelf_floor2",
    "unspecified",
)


# ── JSON schemas ──────────────────────────────────────────────────────────────

WORKSPACE_COUNT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "object_count": {
            "type": "INTEGER",
            "description": (
                "Number of distinct graspable objects currently visible "
                "on the main work table."
            ),
        },
        "reason": {
            "type": "STRING",
            "description": "Brief description of what was counted.",
        },
    },
    "required": ["object_count", "reason"],
}

TASK_PLAN_SCHEMA: dict[str, Any] = {
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

TASK_VERIFICATION_SCHEMA: dict[str, Any] = {
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


# ── Exceptions ────────────────────────────────────────────────────────────────

class GeminiAPIError(RuntimeError):
    """Raised when Gemini generation or validation fails."""


# ── Main API class ────────────────────────────────────────────────────────────

class GeminiAPI:
    """Gemini wrapper for task command parsing and post-pick verification."""

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

    # ── Task parsing ──────────────────────────────────────────────────────────

    def parse_task_command(self, instruction: str) -> dict[str, Any]:
        """Convert one natural-language instruction into a structured task plan.

        Returns a dict with a 'tasks' list, each item containing 'object' and
        'destination'.

        Example input:  "고기 캔과 콜라 캔을 왼쪽 바구니로 옮겨라"
        Example output: {"tasks": [{"object": "meat_can", "destination": "storage_1"},
                                    {"object": "coke_can", "destination": "storage_1"}]}
        """
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("Task instruction is empty.")

        prompt = f"""
You are the command parser for a ROS2 robotic manipulation pipeline.

Convert the instruction into strict JSON containing exactly one task for each
object mentioned.

{OBJECT_DISAMBIGUATION_NOTE}

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
   - "Move the banana to the left storage." -> banana, storage_1
   - "banana"                               -> banana, unspecified

Instruction:
{instruction}
""".strip()

        payload = self._generate_json(
            contents=prompt,
            schema=TASK_PLAN_SCHEMA,
            temperature=0.0,
        )
        return self._validate_task_plan(payload)

    # ── Workspace object counting ─────────────────────────────────────────────

    def count_workspace_objects(self, pil_image: Any) -> dict[str, Any]:
        """Count all distinct graspable objects visible on the workspace table.

        Used for count-based pick verification: compare count before and after
        the pick to determine whether the object was successfully removed,
        without relying on object-specific identification (which can misidentify
        similar-looking objects such as a hammer being called a banana).

        Returns:
            {"object_count": int, "reason": str}
        """
        if not hasattr(pil_image, "size"):
            raise TypeError(
                "count_workspace_objects expects a PIL image with a size attribute."
            )

        prompt = """
You are auditing a robot workspace to count graspable objects.

Count the number of distinct objects currently visible on the main work table
(the flat surface where the robot picks objects from).

Counting rules:
1. Count ONLY objects that a robot gripper could pick up: cans, bottles, boxes,
   fruits, tools, toys, etc.
2. Do NOT count: the robot arm, the gripper, storage baskets, shelves, or the
   table/floor surface itself.
3. Count an object only if more than half of it is visible (not just an edge).
4. Count each physical object exactly once, even if it overlaps another.
5. Objects that have already been placed into a basket or on a shelf do NOT
   count — only objects still on the flat pickup table count.

Return:
- object_count: integer >= 0
- reason: one short sentence listing what you counted (e.g. "1 banana and 1 hammer")
""".strip()

        payload = self._generate_json(
            contents=[prompt, pil_image],
            schema=WORKSPACE_COUNT_SCHEMA,
            temperature=0.0,
        )

        if not isinstance(payload, dict):
            raise GeminiAPIError("Object count response must be a JSON object.")

        count = payload.get("object_count")
        if not isinstance(count, int) or count < 0:
            raise GeminiAPIError(f"Invalid object_count value: {count!r}")

        return {
            "object_count": int(count),
            "reason": str(payload.get("reason", "")).strip(),
        }

    # ── Post-pick verification (legacy — use count-based instead) ─────────────

    def verify_object_removed(
        self,
        pil_image: Any,
        *,
        object_name: str,
        destination: str,
    ) -> dict[str, Any]:
        """Check whether the target object remains in the original pickup workspace.

        The image must be captured after the robot has returned home.  The
        destination area is explicitly excluded so a correctly placed object is
        not treated as a failed pick.

        Returns:
            {
              "present_in_source_workspace": bool,
              "confidence": float,
              "reason": str
            }
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

{OBJECT_DISAMBIGUATION_NOTE}

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

    # ── Internal helpers ──────────────────────────────────────────────────────

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


# ── Module-level helpers ──────────────────────────────────────────────────────

def _normalize_object_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def build_segmentation_prompt(object_name: str) -> str:
    """Build a Gemini segmentation prompt for the given object.

    Includes the visual description (if available) and the disambiguation note
    so the model identifies the correct object when several similar items are
    present on the table.
    """
    display = object_name.replace("_", " ")
    desc = OBJECT_DESCRIPTIONS.get(object_name, "")
    parts: list[str] = [f"Locate and return the bounding box of the {display}."]
    if desc:
        parts.append(desc)
    parts.append(OBJECT_DISAMBIGUATION_NOTE)
    return " ".join(parts)
