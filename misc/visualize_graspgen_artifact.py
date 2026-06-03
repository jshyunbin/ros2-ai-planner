#!/usr/bin/env python3
"""Visualize ros2-ai-planner GraspGen artifacts in the GraspGen viser viewer."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from grasp_gen.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    make_frame,
    visualize_grasp,
    visualize_pointcloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one graspgen_service artifact directory in a Viser web viewer."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Path to one artifacts/graspgen_service/<run>/ directory.",
    )
    parser.add_argument(
        "--gripper-name",
        default="robotiq_2f_140",
        help="Gripper name for grasp wireframe visualization.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the Viser web viewer.",
    )
    parser.add_argument(
        "--max-grasps",
        type=int,
        default=50,
        help="Maximum number of grasps to visualize.",
    )
    parser.add_argument(
        "--show-background",
        action="store_true",
        help="Show background_cloud.npy if present.",
    )
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.asarray(np.load(path), dtype=np.float32)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    if not artifact_dir.is_dir():
        raise SystemExit(f"Artifact directory not found: {artifact_dir}")

    segmented_cloud = load_array(artifact_dir / "segmented_cloud.npy")
    if segmented_cloud is None or segmented_cloud.ndim != 2 or segmented_cloud.shape[1] != 3:
        raise SystemExit(f"Missing or invalid segmented_cloud.npy in {artifact_dir}")

    background_cloud = load_array(artifact_dir / "background_cloud.npy")
    grasps = load_array(artifact_dir / "grasps.npy")
    confidences = load_array(artifact_dir / "confidences.npy")

    result_path = artifact_dir / "result.json"
    payload = {}
    if result_path.is_file():
        with open(result_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

    vis = create_visualizer(port=args.port)
    make_frame(vis, "world", h=0.08, radius=0.003)

    visualize_pointcloud(
        vis,
        "segmented_cloud",
        segmented_cloud,
        color=np.array([0, 220, 0], dtype=np.uint8),
        size=0.004,
    )

    if args.show_background and background_cloud is not None and len(background_cloud) > 0:
        visualize_pointcloud(
            vis,
            "background_cloud",
            background_cloud,
            color=np.array([140, 140, 140], dtype=np.uint8),
            size=0.0025,
        )

    if grasps is not None and grasps.ndim == 3 and grasps.shape[1:] == (4, 4) and len(grasps) > 0:
        if confidences is None or len(confidences) != len(grasps):
            colors = np.tile(np.array([[255, 180, 0]], dtype=np.int32), (len(grasps), 1))
        else:
            conf = np.asarray(confidences, dtype=np.float32).reshape(-1)
            conf_min = float(conf.min())
            conf_max = float(conf.max())
            if conf_max > conf_min:
                normalized = (conf - conf_min) / (conf_max - conf_min)
            else:
                normalized = np.ones_like(conf)
            colors = get_color_from_score(normalized, use_255_scale=True)

        max_grasps = min(int(args.max_grasps), len(grasps))
        for index in range(max_grasps):
            visualize_grasp(
                vis,
                f"grasps/{index:03d}",
                grasps[index],
                color=colors[index],
                gripper_name=args.gripper_name,
                linewidth=1.4,
            )

    print(f"Viewer running at http://localhost:{args.port}")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Segmented points: {len(segmented_cloud)}")
    print(
        f"Background points: {0 if background_cloud is None else len(background_cloud)} "
        f"(shown={args.show_background})"
    )
    print(f"Grasps loaded: {0 if grasps is None else len(grasps)}")
    if payload:
        print(f"Result summary keys: {sorted(payload.keys())}")
    print("Press Ctrl-C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
