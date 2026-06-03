#!/usr/bin/env python3
"""Single script to perform object detection with Gemini and draw annotations.

This script takes an image and a text prompt, asks Gemini for 2D bounding boxes 
in [ymin, xmin, ymax, xmax] format, and outputs an annotated image.
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw
from google import genai
from google.genai import types

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect objects using Gemini and annotate the image.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        default="Detect the meat_can and any other objects of interest.",
        help="Text prompt for what objects to detect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotated_output.png"),
        help="Path to save the annotated output image.",
    )
    return parser.parse_args()

def pick_color(index: int) -> tuple[int, int, int]:
    """Return a visually distinct color for bounding boxes."""
    palette = ["#00d084", "#ffb000", "#ff4d6d", "#5b8cff", "#9b5de5", "#2ec4b6"]
    return ImageColor.getrgb(palette[index % len(palette)])

def main() -> None:
    args = parse_args()

    # 1. Setup API Client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Error: GEMINI_API_KEY environment variable is not set.")
    if not args.image.is_file():
        raise SystemExit(f"Error: Image not found at {args.image}")

    client = genai.Client(api_key=api_key)
    
    # 2. Load the Image
    image = Image.open(args.image).convert("RGB")
    width, height = image.size

    # 3. Define the rigid JSON schema to guarantee [ymin, xmin, ymax, xmax, label] logic
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "box_2d": {
                    "type": "ARRAY",
                    "items": {"type": "INTEGER"},
                    "description": "Bounding box [ymin, xmin, ymax, xmax] scaled strictly from 0 to 1000."
                },
                "label": {
                    "type": "STRING",
                    "description": "Descriptive text label of the detected item."
                }
            },
            "required": ["box_2d", "label"]
        }
    }

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_schema=schema,
        temperature=0.1, # Low temp for more deterministic detection
    )

    # Full prompt combining user request and formatting instructions
    full_prompt = (
        f"{args.prompt}\n"
        "Return a JSON list. For each object, return the label and the 'box_2d' "
        "as an array of exactly 4 integers: [ymin, xmin, ymax, xmax]."
    )

    print(f"Analyzing {args.image.name}...")
    
    # 4. Call Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[full_prompt, image],
        config=config,
    )

    try:
        detections = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse Gemini response: {e}\nRaw output: {response.text}")

    if not detections:
        print("No objects detected.")
        return

    # 5. Draw Annotations
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    for i, item in enumerate(detections):
        box_2d = item.get("box_2d", [])
        label = item.get("label", "unknown")

        if len(box_2d) != 4:
            print(f"Skipping {label}: Invalid bounding box format {box_2d}")
            continue

        # Convert normalized coordinates (0-1000) to actual image pixel coordinates
        ymin_norm, xmin_norm, ymax_norm, xmax_norm = box_2d
        
        xmin = int(round(xmin_norm / 1000.0 * width))
        ymin = int(round(ymin_norm / 1000.0 * height))
        xmax = int(round(xmax_norm / 1000.0 * width))
        ymax = int(round(ymax_norm / 1000.0 * height))

        # Draw Rectangle
        color = pick_color(i)
        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=3)
        
        # Draw Label Text (placed slightly above the bounding box)
        text_y = max(0, ymin - 15)
        
        # Draw a small background rectangle for text readability
        bbox = draw.textbbox((xmin, text_y), label)
        draw.rectangle(bbox, fill=color)
        draw.text((xmin, text_y), label, fill="white")

        print(f"Detected: '{label}' at pixels [ymin:{ymin}, xmin:{xmin}, ymax:{ymax}, xmax:{xmax}]")

    # 6. Save the output
    annotated_image.save(args.output)
    print(f"Saved annotated image to: {args.output.absolute()}")

if __name__ == "__main__":
    main()