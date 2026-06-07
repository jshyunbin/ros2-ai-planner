#!/usr/bin/env python3
"""Gemini detection with confidence threshold.

Same as detect_and_annotate.py but adds confidence to the prompt and
rejects detections below --min-confidence (default 0.8).

Usage:
    python3 misc/detect_and_annotate_confidence.py \
        --image test.jpg \
        --prompt "pick up the strawberry" \
        --min-confidence 0.8 \
        --output result.png
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw
from google import genai
from google.genai import types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects using Gemini with confidence threshold.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="Detect objects.", help="Object to detect.")
    parser.add_argument("--min-confidence", type=float, default=0.8,
                        help="Minimum confidence to accept detection (default: 0.8).")
    parser.add_argument("--output", type=Path, default=Path("annotated_confidence.png"))
    return parser.parse_args()


def pick_color(index: int) -> tuple[int, int, int]:
    palette = ["#00d084", "#ffb000", "#ff4d6d", "#5b8cff", "#9b5de5", "#2ec4b6"]
    return ImageColor.getrgb(palette[index % len(palette)])


def main() -> None:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Error: GEMINI_API_KEY environment variable is not set.")
    if not args.image.is_file():
        raise SystemExit(f"Error: Image not found at {args.image}")

    client = genai.Client(api_key=api_key)
    image = Image.open(args.image).convert("RGB")
    width, height = image.size

    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "box_2d": {
                    "type": "ARRAY",
                    "items": {"type": "INTEGER"},
                    "description": "Bounding box [ymin, xmin, ymax, xmax] scaled strictly from 0 to 1000.",
                },
                "label": {
                    "type": "STRING",
                    "description": "Descriptive text label of the detected item.",
                },
                "confidence": {
                    "type": "NUMBER",
                    "description": "Detection confidence score between 0.0 and 1.0.",
                },
            },
            "required": ["box_2d", "label", "confidence"],
        },
    }

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_schema=schema,
        temperature=0.1,
    )

    # Modified prompt: minimal change from original — adds confidence field
    # and one sentence asking to return [] if not present or below threshold.
    full_prompt = (
        f"{args.prompt}\n"
        "Return a JSON list. For each object, return the label, the 'box_2d' "
        "as an array of exactly 4 integers: [ymin, xmin, ymax, xmax], "
        "and 'confidence' as a float from 0.0 to 1.0. "
        "If the object is not present or confidence is below "
        f"{args.min_confidence:.1f}, return []."
    )

    print(f"Prompt:\n{full_prompt}\n")
    print(f"Analyzing {args.image.name} (min_confidence={args.min_confidence})...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[full_prompt, image],
        config=config,
    )

    try:
        detections = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse Gemini response: {e}\nRaw: {response.text}")

    print(f"Raw response: {json.dumps(detections, indent=2)}")

    if not detections:
        print(f"No detections (Gemini returned []) — object likely not present or confidence < {args.min_confidence}.")
        return

    # Filter by confidence
    accepted = [d for d in detections if float(d.get("confidence", 1.0)) >= args.min_confidence]
    rejected = [d for d in detections if float(d.get("confidence", 1.0)) < args.min_confidence]

    for d in rejected:
        print(f"REJECTED '{d.get('label')}': confidence={d.get('confidence'):.2f} < {args.min_confidence}")

    if not accepted:
        print("All detections rejected by confidence threshold.")
        return

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for i, item in enumerate(accepted):
        box_2d = item.get("box_2d", [])
        label = item.get("label", "unknown")
        confidence = float(item.get("confidence", 0.0))

        if len(box_2d) != 4:
            print(f"Skipping '{label}': invalid box {box_2d}")
            continue

        ymin_n, xmin_n, ymax_n, xmax_n = box_2d
        xmin = int(round(xmin_n / 1000.0 * width))
        ymin = int(round(ymin_n / 1000.0 * height))
        xmax = int(round(xmax_n / 1000.0 * width))
        ymax = int(round(ymax_n / 1000.0 * height))

        color = pick_color(i)
        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=3)

        text = f"{label} {confidence:.2f}"
        text_y = max(0, ymin - 15)
        bbox = draw.textbbox((xmin, text_y), text)
        draw.rectangle(bbox, fill=color)
        draw.text((xmin, text_y), text, fill="white")

        print(f"ACCEPTED '{label}': confidence={confidence:.2f} bbox=[{ymin},{xmin},{ymax},{xmax}]")

    annotated.save(args.output)
    print(f"\nSaved to: {args.output.absolute()}")


if __name__ == "__main__":
    main()
