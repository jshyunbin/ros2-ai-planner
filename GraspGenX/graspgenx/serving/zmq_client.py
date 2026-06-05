# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin ZMQ REQ client for :class:`GraspGenXZMQServer`.

This module deliberately has no dependency on torch, the model weights, or any
gripper asset — it's a pure msgpack/ZMQ wire-protocol shim. See
:mod:`graspgenx.serving.zmq_server` for the protocol.
"""

from __future__ import annotations

from typing import Optional, Tuple

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from graspgenx.utils.logging_config import get_logger

msgpack_numpy.patch()

logger = get_logger(__name__)


class GraspGenXClient:
    """ZMQ REQ client that round-trips msgpack payloads to a GraspGenX server.

    Usage::

        with GraspGenXClient(host="localhost", port=5556) as client:
            print(client.server_metadata)
            grasps, confidences = client.infer(
                point_cloud=xyz, gripper_name="franka_panda",
            )

    Args:
        host: Server hostname (default ``localhost``).
        port: Server port (default ``5556``).
        timeout_ms: Per-request send/recv timeout in milliseconds. ``None``
            disables timeouts (the request blocks until the server replies).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        timeout_ms: Optional[int] = 60_000,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None
        self._metadata_cache: Optional[dict] = None

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def __enter__(self) -> "GraspGenXClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self._sock is not None:
            return
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        if self.timeout_ms is not None:
            self._sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_ms))
            self._sock.setsockopt(zmq.SNDTIMEO, int(self.timeout_ms))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.address)
        logger.info("Connected to GraspGenX server at %s", self.address)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None

    def _request(self, payload: dict) -> dict:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        try:
            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._sock.recv()
        except zmq.error.Again as exc:
            # Socket is in a bad state after a timeout — reset so callers can retry.
            self.close()
            raise TimeoutError(
                f"GraspGenX server at {self.address} did not respond within {self.timeout_ms} ms"
            ) from exc
        response = msgpack.unpackb(raw, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GraspGenX server error: {response['error']}")
        return response

    @property
    def server_metadata(self) -> dict:
        """Cached metadata response. Re-fetched once per client lifetime."""
        if self._metadata_cache is None:
            self._metadata_cache = self._request({"action": "metadata"})
        return self._metadata_cache

    def health(self) -> dict:
        return self._request({"action": "health"})

    def infer(
        self,
        point_cloud: np.ndarray,
        gripper_name: Optional[str] = None,
        num_grasps: int = 200,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Send a point cloud + gripper to the server, return ranked grasps.

        Returns:
            (grasps, confidences) — grasps is (K, 4, 4) float32, confidences
            is (K,) float32. Both arrays may be empty if the model produced
            no above-threshold grasps.
        """
        pc = np.asarray(point_cloud, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"point_cloud must be (N, 3); got {pc.shape}")

        payload = {
            "action": "infer",
            "point_cloud": pc,
            "num_grasps": int(num_grasps),
            "grasp_threshold": float(grasp_threshold),
            "topk_num_grasps": int(topk_num_grasps),
        }
        if gripper_name is not None:
            payload["gripper_name"] = gripper_name

        response = self._request(payload)
        grasps = np.asarray(response["grasps"], dtype=np.float32)
        confidences = np.asarray(response["confidences"], dtype=np.float32)
        return grasps, confidences
