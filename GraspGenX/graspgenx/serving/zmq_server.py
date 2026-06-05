# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZMQ REQ/REP server that wraps :class:`GraspGenXSampler` for remote inference.

Wire protocol (msgpack with ``msgpack_numpy.patch()`` so numpy arrays travel
natively):

* ``{"action": "health"}`` → ``{"status": "ok"}``
* ``{"action": "metadata"}`` →
  ``{"default_gripper": str|None, "loaded_grippers": [str], "model": {...}}``
* ``{"action": "infer", "point_cloud": (N,3) float32,
       "gripper_name": str (optional, falls back to default),
       "num_grasps": int = 200,
       "grasp_threshold": float = -1.0,
       "topk_num_grasps": int = 100}`` →
  ``{"grasps": (K,4,4) float32, "confidences": (K,) float32,
     "gripper_name": str, "timing": {"infer_ms": float}}``

Any unhandled error is returned as ``{"error": str(exc)}`` — the client raises.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.utils.logging_config import get_logger

msgpack_numpy.patch()

logger = get_logger(__name__)


class GraspGenXZMQServer:
    """ZMQ REQ/REP wrapper around :class:`GraspGenXSampler`.

    Samplers are loaded lazily on the first ``infer`` request for a given
    gripper and cached in-process — so the first request per gripper pays the
    model-init cost, every subsequent request is hot.

    Args:
        config_path: Path to the checkpoint root containing ``gen/`` and
            ``dis/`` subdirectories (each with ``config.yaml`` + ``epoch_*.pth``).
            For backward compatibility a path that ends in ``config.yaml``
            inside a ``gen/`` directory is accepted — its grandparent is used
            as the root.
        assets_dir: Root directory containing ``x_grippers/`` and
            ``proc_grippers/`` subdirectories. Passed through to each sampler.
        host: ZMQ bind address (default ``0.0.0.0``).
        port: ZMQ bind port (default ``5556``).
        default_gripper: If set, pre-load this gripper at startup and use it
            when clients omit ``gripper_name``.
    """

    def __init__(
        self,
        config_path: str,
        assets_dir: str,
        host: str = "0.0.0.0",
        port: int = 5556,
        default_gripper: Optional[str] = None,
    ) -> None:
        self.assets_dir = str(Path(assets_dir).expanduser().resolve())
        self.host = host
        self.port = port
        self.default_gripper = default_gripper

        self.cfg = self._load_cfg(config_path)
        self._samplers: Dict[str, GraspGenXSampler] = {}
        self._samplers_lock = threading.Lock()

        if default_gripper:
            logger.info("Pre-loading default gripper: %s", default_gripper)
            self._get_sampler(default_gripper)

    @staticmethod
    def _load_cfg(config_path: str):
        """Resolve a user-supplied path to a merged gen+dis config."""
        p = Path(config_path).expanduser().resolve()
        if p.is_file() and p.name == "config.yaml" and p.parent.name == "gen":
            root = p.parent.parent
        elif p.is_dir():
            root = p
        else:
            raise FileNotFoundError(
                f"--config must be a checkpoint root containing gen/ and dis/ "
                f"subdirs (or path to gen/config.yaml). Got: {config_path}"
            )
        gen_dir = root / "gen"
        dis_dir = root / "dis"
        if not gen_dir.is_dir() or not dis_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint root {root} must contain gen/ and dis/ subdirs."
            )
        return load_model_cfg(str(gen_dir), str(dis_dir))

    def _get_sampler(self, gripper_name: str) -> GraspGenXSampler:
        with self._samplers_lock:
            sampler = self._samplers.get(gripper_name)
            if sampler is None:
                logger.info("Loading sampler for gripper: %s", gripper_name)
                sampler = GraspGenXSampler(
                    self.cfg, gripper_name, assets_dir=self.assets_dir
                )
                self._samplers[gripper_name] = sampler
            return sampler

    def _handle_metadata(self) -> dict:
        diff = self.cfg.diffusion
        dis = self.cfg.discriminator
        return {
            "default_gripper": self.default_gripper,
            "loaded_grippers": sorted(self._samplers.keys()),
            "model": {
                "generator_backbone": str(diff.object_backbone),
                "discriminator_backbone": str(dis.object_backbone),
                "grasp_repr": str(diff.grasp_repr),
                "num_diffusion_iters_eval": int(diff.num_diffusion_iters_eval),
            },
            "assets_dir": self.assets_dir,
        }

    def _handle_infer(self, request: dict) -> dict:
        gripper_name = request.get("gripper_name") or self.default_gripper
        if not gripper_name:
            raise ValueError(
                "Request omitted 'gripper_name' and no default_gripper is set."
            )
        point_cloud = request.get("point_cloud")
        if point_cloud is None:
            raise ValueError("Request is missing 'point_cloud'.")
        pc = np.asarray(point_cloud, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"point_cloud must be (N, 3); got {pc.shape}")

        num_grasps = int(request.get("num_grasps", 200))
        grasp_threshold = float(request.get("grasp_threshold", -1.0))
        topk_num_grasps = int(request.get("topk_num_grasps", 100))

        sampler = self._get_sampler(gripper_name)

        t0 = time.monotonic()
        grasps, confidences = GraspGenXSampler.run_inference(
            pc,
            sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
        )
        infer_ms = (time.monotonic() - t0) * 1000.0

        grasps_np = (
            grasps.detach().cpu().numpy().astype(np.float32)
            if len(grasps)
            else np.zeros((0, 4, 4), dtype=np.float32)
        )
        conf_np = (
            confidences.detach().cpu().numpy().astype(np.float32)
            if len(confidences)
            else np.zeros((0,), dtype=np.float32)
        )

        return {
            "grasps": grasps_np,
            "confidences": conf_np,
            "gripper_name": gripper_name,
            "timing": {"infer_ms": float(infer_ms)},
        }

    def _dispatch(self, request: dict) -> dict:
        action = request.get("action")
        if action == "health":
            return {"status": "ok"}
        if action == "metadata":
            return self._handle_metadata()
        if action == "infer":
            return self._handle_infer(request)
        raise ValueError(f"Unknown action: {action!r}")

    def serve_forever(self) -> None:
        """Bind the REP socket and serve requests until interrupted."""
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        addr = f"tcp://{self.host}:{self.port}"
        sock.bind(addr)
        logger.info("GraspGenX ZMQ server listening on %s", addr)
        try:
            while True:
                raw = sock.recv()
                try:
                    request = msgpack.unpackb(raw, raw=False)
                    response = self._dispatch(request)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Request failed: %s", exc)
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                sock.send(msgpack.packb(response, use_bin_type=True))
        except KeyboardInterrupt:
            logger.info("Shutting down ZMQ server (KeyboardInterrupt).")
        finally:
            sock.close(linger=0)
