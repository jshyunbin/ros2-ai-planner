import numpy as np

from team_8.segmentation_utils import (
    PromptPoint,
    build_overlay_image,
    compute_centroid,
    compute_mask_roi,
    depth_to_masked_points,
    resize_for_api,
    stable_downsample,
)


def test_depth_to_masked_points_and_centroid():
    depth = np.full((3, 3), 0.5, dtype=np.float32)
    depth[1, 1] = 0.4

    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    object_points, background_points = depth_to_masked_points(
        depth,
        mask,
        fx=1.0,
        fy=1.0,
        cx=1.0,
        cy=1.0,
        min_depth_m=0.1,
        max_depth_m=1.0,
    )

    assert object_points.shape == (1, 3)
    assert background_points.shape == (8, 3)
    # The masked pixel is at the principal point, so x=y=0 and z=depth.
    assert np.allclose(object_points[0], [0.0, 0.0, 0.4])
    centroid = compute_centroid(object_points, surface_band_m=0.02)
    assert np.allclose(centroid, object_points[0])


def test_compute_mask_roi():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:9, 2:8] = True
    assert compute_mask_roi(mask) == (2, 2, 7, 8)


def test_resize_for_api_keeps_small_images():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    resized, scale_x, scale_y = resize_for_api(image, max_dimension=64)
    assert resized.shape == image.shape
    assert scale_x == 1.0 and scale_y == 1.0


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
