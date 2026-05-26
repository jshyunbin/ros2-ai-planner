import numpy as np

from pipeline_orchestrator.segmentation_utils import (
    PromptPoint,
    build_overlay_image,
    compute_centroid,
    compute_mask_roi,
    parse_sam3_polygons,
    rasterize_polygons,
    select_masked_points,
    stable_downsample,
)


def test_parse_sam3_polygons_reads_prompt_results_shape():
    payload = {
        "prompt_results": [
            {
                "predictions": [
                    {
                        "label": "banana",
                        "confidence": 0.97,
                        "masks": [[[1, 1], [5, 1], [5, 5], [1, 5]]],
                    }
                ]
            }
        ]
    }

    polygons = parse_sam3_polygons(payload)

    assert len(polygons) == 1
    assert polygons[0].shape == (4, 2)


def test_rasterize_polygons_and_roi():
    polygon = np.asarray([[2, 2], [7, 2], [7, 8], [2, 8]], dtype=np.float32)
    mask = rasterize_polygons([polygon], width=10, height=10)

    assert mask.sum() > 0
    assert compute_mask_roi(mask) == (2, 2, 7, 8)


def test_select_masked_points_and_centroid():
    xyz = np.zeros((3, 3, 3), dtype=np.float32)
    xyz[:, :, 2] = 0.5
    xyz[1, 1] = np.array([0.2, 0.3, 0.4], dtype=np.float32)

    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    object_points, background_points = select_masked_points(
        xyz,
        mask,
        min_depth_m=0.1,
        max_depth_m=1.0,
    )

    assert object_points.shape == (1, 3)
    assert background_points.shape == (8, 3)
    centroid = compute_centroid(object_points, surface_band_m=0.02)
    assert np.allclose(centroid, object_points[0])


def test_stable_downsample_preserves_order():
    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    downsampled = stable_downsample(points, 4)

    assert downsampled.shape == (4, 3)
    assert np.allclose(downsampled[0], points[0])
    assert np.allclose(downsampled[-1], points[-1])


def test_build_overlay_image_returns_same_shape():
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 4:10] = True

    overlay = build_overlay_image(
        image,
        mask,
        [PromptPoint(x=6, y=6, positive=True)],
        "target",
    )

    assert overlay.shape == image.shape
