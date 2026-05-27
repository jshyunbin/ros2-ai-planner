#!/usr/bin/env python3
"""Visual check for Gemini bbox + Ultralytics SAM2 segmentation."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw
from google import genai
from google.genai import types
from ultralytics import SAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect an object with Gemini, segment it with Ultralytics SAM2, and save an overlay."
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--object",
        dest="object_name",
        required=True,
        help="Object name to detect, used to build the fixed Gemini prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sam2_annotated_output.png"),
        help="Path to save the annotated output image.",
    )
    parser.add_argument(
        "--model",
        default="sam2_t.pt",
        help="Ultralytics SAM2 checkpoint name or local path.",
    )
    return parser.parse_args()


def pick_color(index: int) -> tuple[int, int, int]:
    palette = ["#00d084", "#ffb000", "#ff4d6d", "#5b8cff", "#9b5de5", "#2ec4b6"]
    return ImageColor.getrgb(palette[index % len(palette)])


def scale_box_2d(box_2d: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    if len(box_2d) != 4:
        raise ValueError(f"Invalid box_2d: {box_2d}")

    ymin_norm, xmin_norm, ymax_norm, xmax_norm = [int(v) for v in box_2d]
    xmin = int(round(np.clip(xmin_norm, 0, 1000) / 1000.0 * width))
    ymin = int(round(np.clip(ymin_norm, 0, 1000) / 1000.0 * height))
    xmax = int(round(np.clip(xmax_norm, 0, 1000) / 1000.0 * width))
    ymax = int(round(np.clip(ymax_norm, 0, 1000) / 1000.0 * height))

    xmin = int(np.clip(xmin, 0, width - 1))
    ymin = int(np.clip(ymin, 0, height - 1))
    xmax = int(np.clip(xmax, 0, width - 1))
    ymax = int(np.clip(ymax, 0, height - 1))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Degenerate scaled box: {[xmin, ymin, xmax, ymax]}")
    return xmin, ymin, xmax, ymax


def gemini_detect(image: Image.Image, prompt: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Error: GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
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
            },
            "required": ["box_2d", "label"],
        },
    }

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_schema=schema,
        temperature=0.1,
    )
    full_prompt = (
        f"{prompt}\n"
        "Return a JSON list. For each object, return the label and the 'box_2d' "
        "as an array of exactly 4 integers: [ymin, xmin, ymax, xmax]."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[full_prompt, image],
        config=config,
    )
    detections = json.loads(response.text)
    if not detections:
        raise SystemExit("No objects detected.")
    return detections[0]


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Error: Image not found at {args.image}")

    image = Image.open(args.image).convert("RGB")
    width, height = image.size

    prompt = f"Detect a {args.object_name} and return [ymin, xmin, ymax, xmax, label]"
    print(f"Analyzing {args.image.name}...")
    detection = gemini_detect(image, prompt)
    label = str(detection.get("label", "")).strip() or "target"
    box_2d = detection.get("box_2d", [])
    xmin, ymin, xmax, ymax = scale_box_2d(box_2d, width, height)

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    model = SAM(args.model)
    results = model(image_bgr, bboxes=[xmin, ymin, xmax, ymax])
    masks = results[0].masks
    if masks is None or masks.data is None or len(masks.data) == 0:
        raise SystemExit("SAM2 returned no mask.")

    mask = masks.data[0].cpu().numpy() > 0

    annotated = image_bgr.copy()
    color = pick_color(0)
    tint = np.zeros_like(annotated)
    tint[:, :] = (color[2], color[1], color[0])
    annotated[mask] = cv2.addWeighted(annotated, 0.55, tint, 0.45, 0.0)[mask]

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 255, 255), 2)
    cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated_pil)
    text = f"{label}"
    text_y = max(0, ymin - 15)
    text_box = draw.textbbox((xmin, text_y), text)
    draw.rectangle(text_box, fill=color)
    draw.text((xmin, text_y), text, fill="white")

    annotated_pil.save(args.output)
    print(f"Label: {label}")
    print(f"Gemini box_2d: {box_2d}")
    print(f"Scaled bbox xyxy: {[xmin, ymin, xmax, ymax]}")
    print(f"Mask shape: {tuple(mask.shape)}")
    print(f"Saved annotated image to: {args.output.absolute()}")


if __name__ == "__main__":
    main()
