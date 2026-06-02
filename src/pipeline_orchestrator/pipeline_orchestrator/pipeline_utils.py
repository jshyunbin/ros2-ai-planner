"""Small shared helpers used across the pipeline_orchestrator nodes.

Kept dependency-light so it can be imported by every node (env parsing, bool
coercion, and the XYZ PointCloud2 builder).
"""

import os

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ('0', 'false', 'no', 'off')


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return float(default)
    return float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return int(default)
    return int(raw)


def make_xyz_cloud(points: np.ndarray, frame_id: str, stamp=None) -> PointCloud2:
    """Build a dense XYZ ``sensor_msgs/PointCloud2`` from an (N, 3+) array.

    ``stamp`` is an optional ``builtin_interfaces/Time``; when supplied it is
    written to the cloud header so downstream consumers can correlate the cloud
    with the sensor frame it was built from.
    """
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id)
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(xyz)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(xyz)
    msg.data = xyz.tobytes()
    msg.is_dense = True
    return msg
