"""Single source of truth for the gripper assets dir and TCP z-offset.

Both the grasp-generation stage (``graspgen_service``) and the motion-planning
stage (``curobo``) resolve the tool0->TCP depth from here, so the two stages
always agree on where tool0 sits relative to the GraspGen TCP.

IMPORTANT: the value is sourced the SAME way the GraspGenX inference server
loads a gripper — from the ``gripper_descriptions`` assets
(``<assets_dir>/x_grippers/<gripper>/config.json``), NOT from
``graspgenx.robot.get_gripper_depth`` (which globs ``graspgenx/config/grippers``,
a directory that is empty and never populated by the asset clone — that API
raises for every gripper in the serving deployment).

TCP-depth resolution precedence:
  1. ``PIPELINE_GRIPPER_TCP_Z_OFFSET_M`` env override (explicit, wins; no assets
     needed — useful in CI / asset-less environments).
  2. ``<assets_dir>/x_grippers/<gripper>/config.json`` -> ``fingertip[-1]``,
     mirroring ``graspgenx.x_grippers.get_gripper_info`` (``XGripperInfo.depth``).

There is no hardcoded fallback: if neither source is available the resolver
raises so the pipeline fails loudly rather than planning with a wrong depth.
"""

import json
import os
from pathlib import Path

# The physical UR5 tool is a Robotiq 2F-85.
DEFAULT_GRIPPER_NAME = "robotiq_2f_85"


def gripper_assets_dir() -> Path:
    """Directory that contains ``x_grippers/`` (matches the server --assets_dir).

    Precedence: ``GRASPGENX_ASSETS_DIR`` env (the same var the server script
    sets) -> GraspGenX's auto-resolved gripper_descriptions assets. ``graspgenx``
    is imported lazily so this module does not eagerly pull it in.
    """
    env = os.environ.get("GRASPGENX_ASSETS_DIR")
    if env:
        return Path(env)
    # get_gripper_descriptions_assets() returns ``.../assets/x_grippers``; the
    # server's --assets_dir is its parent (the dir holding x_grippers/).
    from graspgenx import get_gripper_descriptions_assets

    return Path(get_gripper_descriptions_assets()).parent


def resolve_gripper_tcp_z_offset(gripper_name: str = DEFAULT_GRIPPER_NAME):
    """Resolve the gripper TCP z-offset in meters.

    Returns ``(offset_m, source_str)``. Raises ``RuntimeError`` if the value
    cannot be sourced from the env override or the gripper_descriptions assets.
    """
    env_val = os.environ.get("PIPELINE_GRIPPER_TCP_Z_OFFSET_M")
    if env_val is not None:
        return float(env_val), "env:PIPELINE_GRIPPER_TCP_Z_OFFSET_M"
    try:
        cfg_path = gripper_assets_dir() / "x_grippers" / gripper_name / "config.json"
        with open(cfg_path, "r") as f:
            config = json.load(f)
        # Mirrors graspgenx.x_grippers.get_gripper_info: XGripperInfo.depth =
        # config["fingertip"][-1] — the same TCP depth the GraspGenX sampler uses.
        return float(config["fingertip"][-1]), f"gripper_descriptions:{cfg_path}"
    except Exception as exc:  # assets missing / unreadable / schema mismatch
        raise RuntimeError(
            f"Cannot resolve gripper TCP z-offset for {gripper_name}: set "
            "PIPELINE_GRIPPER_TCP_Z_OFFSET_M, or ensure the GraspGenX "
            f"gripper_descriptions assets exist (x_grippers/{gripper_name}/"
            f"config.json). ({type(exc).__name__}: {exc})"
        ) from exc
