"""Thin ZMQ client for the GraspGenX inference server.

Wraps the GraspGenX wire protocol (msgpack over ZMQ REQ/REP) with the same
interface that graspgen_service.py expects, so the rest of the pipeline needs
no changes beyond this file.

GraspGenX differences from the old GraspGen client:
  - infer() accepts gripper_name (passed at call time, no per-model config)
  - No min_grasps / max_tries / remove_outliers parameters
  - server_metadata returns loaded_grippers list instead of gripper_name
"""

import time

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()

# Default gripper used when caller does not specify one.
_DEFAULT_GRIPPER = 'robotiq_2f_85'


class GraspGenClient:
    """ZMQ REQ client for the GraspGenX inference server.

    The class name is kept as GraspGenClient so graspgen_service.py imports
    remain unchanged.
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5556,
        timeout_ms: int = 60000,
        wait_for_server: bool = True,
        retry_interval_s: float = 2.0,
    ) -> None:
        self._addr = f'tcp://{host}:{port}'
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context()
        self._socket = None
        self._server_metadata = None

        if wait_for_server:
            self._wait_for_server(retry_interval_s)

    # ── connection ────────────────────────────────────────────────────────────

    def _create_socket(self):
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._addr)
        return sock

    def _ensure_connected(self) -> None:
        if self._socket is None:
            self._socket = self._create_socket()

    def _request(self, payload: dict) -> dict:
        self._ensure_connected()
        self._socket.send(msgpack.packb(payload, use_bin_type=True))
        raw = self._socket.recv()
        response = msgpack.unpackb(raw, raw=False)
        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f'GraspGenX server error: {response["error"]}')
        return response

    def _wait_for_server(self, retry_interval_s: float) -> None:
        while True:
            try:
                self._socket = self._create_socket()
                self._server_metadata = self._request({'action': 'metadata'})
                return
            except (zmq.error.Again, zmq.error.ZMQError):
                if self._socket is not None:
                    self._socket.close()
                    self._socket = None
                time.sleep(retry_interval_s)

    @property
    def server_metadata(self):
        return self._server_metadata

    # ── inference ─────────────────────────────────────────────────────────────

    def infer(
        self,
        point_cloud: np.ndarray,
        *,
        gripper_name: str = _DEFAULT_GRIPPER,
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = 100,
        # Legacy kwargs accepted but ignored (GraspGenX does not support them).
        min_grasps: int = 0,
        max_tries: int = 0,
        remove_outliers: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Send a point cloud to the server and return (grasps, confidences).

        Args:
            point_cloud: (N, 3) float32 array in base_link frame.
            gripper_name: GraspGenX gripper asset name (default: robotiq_2f_85).
            grasp_threshold: Minimum discriminator score (-1 = no threshold).
            num_grasps: Number of diffusion samples to draw.
            topk_num_grasps: Return at most this many grasps ranked by score.

        Returns:
            grasps: (K, 4, 4) float32 homogeneous TCP transforms.
            confidences: (K,) float32 discriminator scores.
        """
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(
                f'point_cloud must be (N, 3), got {point_cloud.shape}')

        payload = {
            'action': 'infer',
            'point_cloud': point_cloud,
            'gripper_name': gripper_name,
            'grasp_threshold': float(grasp_threshold),
            'num_grasps': int(num_grasps),
            'topk_num_grasps': int(topk_num_grasps),
        }
        response = self._request(payload)
        grasps = np.asarray(response['grasps'], dtype=np.float32)
        confidences = np.asarray(response['confidences'], dtype=np.float32)
        return grasps, confidences

    # ── cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._ctx.term()
