"""Small shared helpers used across the team_8 nodes.

Kept dependency-light so it can be imported by every node (env parsing, bool
coercion, the XYZ PointCloud2 builder, and the grasp-row pose /
rotation-to-quaternion helpers).

Timing utilities
----------------
``TimingLogger``    — wraps a rclpy logger, prepending ``[+Δt | total]`` to
                      every message so bottlenecks are visible in docker logs.
``TimedLoggerMixin``— mixin for rclpy Node subclasses; overrides
                      ``get_logger()`` to return a ``TimingLogger`` without
                      touching any existing log call-sites.

Usage::

    from team_8.pipeline_utils import TimedLoggerMixin

    class MyNode(TimedLoggerMixin, Node):
        ...

    # All existing self.get_logger().info(...) calls now emit timing info.
"""

# graspgenX outputs the tool0 frame directly (translation = where tool0 should go).
# cuRobo also targets tool0. So no backing-off offset is needed between graspgenX
# output and cuRobo input.  Set to 0.0.
# (Legacy value was 0.085m for old GraspGen which output fingertip contact points.)
ROBOTIQ_2F_85_TCP_Z_OFFSET = 0.0  # metres

import math
import os

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

try:  # pragma: no cover - geometry_msgs only present in the ROS runtime
    from geometry_msgs.msg import Pose
except ImportError:  # pragma: no cover - import-only test fallback
    Pose = None


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


def cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Decode a ``sensor_msgs/PointCloud2`` into an (N, 3) float32 array.

    Assumes X, Y, Z are the first three float32 fields (byte offsets 0/4/8);
    any trailing fields in ``point_step`` are ignored. Every producer in this
    package (``make_xyz_cloud``, the segmentation clouds, the TSDF voxels)
    satisfies that layout.
    """
    if msg.width * msg.height == 0:
        return np.empty((0, 3), dtype=np.float32)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    raw = raw.reshape(msg.height * msg.width, msg.point_step)
    # .copy(): the column slice is non-contiguous; make it contiguous before .view().
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    return np.nan_to_num(xyz, nan=0.0)


def quat_from_rotation_matrix(rotation) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    r00, r01, r02 = [float(v) for v in rotation[0]]
    r10, r11, r12 = [float(v) for v in rotation[1]]
    r20, r21, r22 = [float(v) for v in rotation[2]]
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s)
    if r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        return ((r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s)
    if r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        return ((r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s)
    s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
    return ((r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s)


def pose_from_grasp_row(row: dict):
    """Build a geometry_msgs/Pose from a GraspGen rank row dict.

    Returns None when geometry_msgs is unavailable or the row lacks a valid
    3-vector translation / 3x3 rotation matrix.
    """
    if Pose is None:
        return None
    translation = row.get("translation")
    rotation = row.get("rotation_matrix")
    if translation is None or rotation is None:
        return None
    if len(translation) != 3 or len(rotation) != 3:
        return None
    w, x, y, z = quat_from_rotation_matrix(rotation)
    pose = Pose()
    pose.position.x = float(translation[0])
    pose.position.y = float(translation[1])
    pose.position.z = float(translation[2])
    pose.orientation.w = w
    pose.orientation.x = x
    pose.orientation.y = y
    pose.orientation.z = z
    return pose


# ── Timing logger ─────────────────────────────────────────────────────────────

import threading as _threading
import time as _time


class TimingLogger:
    """Thin wrapper around a rclpy logger that prepends elapsed-time tags.

    Every log line becomes::

        [+  0.123s |   4.567s] original message
         ^^^^^^^^^   ^^^^^^^^
         delta from  total from
         last call   node start

    The delta quickly shows which step is slow; the total gives an absolute
    timeline for cross-node comparison.

    Thread-safe: a single lock serialises ``_last_t`` updates so concurrent
    callbacks don't produce garbled timestamps.
    """

    def __init__(self, ros_logger):
        self._log = ros_logger
        self._t0 = _time.monotonic()
        self._last_t = self._t0
        self._lock = _threading.Lock()

    def _tag(self) -> str:
        now = _time.monotonic()
        with self._lock:
            delta = now - self._last_t
            total = now - self._t0
            self._last_t = now
        return f'[+{delta:6.3f}s |{total:7.3f}s] '

    def debug(self, msg, *args, **kwargs):
        self._log.debug(self._tag() + str(msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log.info(self._tag() + str(msg), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log.warning(self._tag() + str(msg), *args, **kwargs)

    # rclpy exposes both .warn() and .warning() — support both
    def warn(self, msg, *args, **kwargs):
        self._log.warning(self._tag() + str(msg), *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log.error(self._tag() + str(msg), *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        self._log.fatal(self._tag() + str(msg), *args, **kwargs)

    # Pass through any attribute access not defined here (e.g. set_level)
    def __getattr__(self, name):
        return getattr(self._log, name)


class TimedLoggerMixin:
    """Mixin for rclpy Node subclasses that makes get_logger() return a
    ``TimingLogger`` without requiring any changes to existing log call-sites.

    Usage::

        class MyNode(TimedLoggerMixin, Node):
            ...

    MRO note: ``TimedLoggerMixin`` must appear *before* ``Node`` in the base
    list so that its ``get_logger()`` shadows ``Node.get_logger()``.
    """

    def get_logger(self):  # type: ignore[override]
        # Use __dict__ directly to bypass any __setattr__ override in rclpy Node
        # and avoid AttributeError from __slots__-based implementations.
        tl = self.__dict__.get('_timed_logger')
        if tl is None:
            tl = TimingLogger(super().get_logger())
            self.__dict__['_timed_logger'] = tl
        return tl
