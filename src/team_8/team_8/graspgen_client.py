import time
from typing import Optional

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()


class GraspGenClient:
    """Minimal ZMQ client for the standalone GraspGen inference server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        timeout_ms: int = 60000,
        wait_for_server: bool = True,
        retry_interval_s: float = 2.0,
    ) -> None:
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context()
        self._socket = None
        self._server_metadata = None

        if wait_for_server:
            self._wait_for_server(retry_interval_s)

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
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def _wait_for_server(self, retry_interval_s: float) -> None:
        while True:
            try:
                self._socket = self._create_socket()
                self._server_metadata = self._request({"action": "metadata"})
                return
            except (zmq.error.Again, zmq.error.ZMQError):
                if self._socket is not None:
                    self._socket.close()
                    self._socket = None
                time.sleep(retry_interval_s)

    @property
    def server_metadata(self):
        return self._server_metadata

    def infer(
        self,
        point_cloud: np.ndarray,
        gripper_name: Optional[str] = None,
        *,
        num_grasps: int = 200,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Send a point cloud (+ optional gripper) to the GraspGenX server.

        The GraspGenX server dropped the old ``min_grasps``/``max_tries``/
        ``remove_outliers`` retry fields; this signature mirrors GraspGenX's
        ``infer(point_cloud, gripper_name, num_grasps, grasp_threshold,
        topk_num_grasps)``. When ``gripper_name`` is None the server uses its
        configured ``--default_gripper``.
        """
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(f"point_cloud must be (N, 3), got {point_cloud.shape}")

        payload = {
            "action": "infer",
            "point_cloud": point_cloud,
            "num_grasps": num_grasps,
            "grasp_threshold": grasp_threshold,
            "topk_num_grasps": topk_num_grasps,
        }
        if gripper_name is not None:
            payload["gripper_name"] = gripper_name
        response = self._request(payload)
        grasps = np.asarray(response["grasps"], dtype=np.float32)
        confidences = np.asarray(response["confidences"], dtype=np.float32)
        return grasps, confidences

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._ctx.term()

