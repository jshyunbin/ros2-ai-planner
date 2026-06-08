"""Unit tests for pose_probe's pure helpers (no live ROS graph required)."""

import math

import pytest

from team_8.pose_probe import (
    orientation_error_deg,
    parse_pose_input,
    position_error_m,
)


# --- parse_pose_input ---------------------------------------------------------

def test_parse_seven_numbers_returns_pose_floats():
    kind, value = parse_pose_input("0.75 -0.3 0.78 -0.5 0.5 -0.5 0.5")
    assert kind == "pose"
    assert value == pytest.approx([0.75, -0.3, 0.78, -0.5, 0.5, -0.5, 0.5])


def test_parse_seven_numbers_accepts_commas():
    kind, value = parse_pose_input("0,0,0.37,1,0,0,0")
    assert kind == "pose"
    assert value == pytest.approx([0.0, 0.0, 0.37, 1.0, 0.0, 0.0, 0.0])


def test_parse_single_token_is_key():
    kind, value = parse_pose_input("bookshelf")
    assert kind == "key"
    assert value == "bookshelf"


def test_parse_empty_returns_none():
    assert parse_pose_input("   ") is None


def test_parse_wrong_number_count_is_error():
    kind, _ = parse_pose_input("0.1 0.2 0.3")
    assert kind == "error"


def test_parse_seven_nonnumeric_tokens_is_error():
    # Seven tokens but not all numbers -> not a valid raw pose.
    kind, _ = parse_pose_input("a b c d e f g")
    assert kind == "error"


# --- position_error_m ---------------------------------------------------------

def test_position_error_zero_when_identical():
    assert position_error_m([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == pytest.approx(0.0)


def test_position_error_euclidean():
    assert position_error_m([0.0, 0.0, 0.0], [0.03, 0.04, 0.0]) == pytest.approx(0.05)


# --- orientation_error_deg ----------------------------------------------------

def test_orientation_error_zero_when_identical():
    q = [0.0, 0.0, 0.0, 1.0]
    assert orientation_error_deg(q, q) == pytest.approx(0.0, abs=1e-6)


def test_orientation_error_sign_insensitive():
    # q and -q represent the same orientation -> zero error.
    q = [0.0, 0.0, 0.0, 1.0]
    neg = [0.0, 0.0, 0.0, -1.0]
    assert orientation_error_deg(q, neg) == pytest.approx(0.0, abs=1e-6)


def test_orientation_error_ninety_degrees_about_z():
    # 90 deg rotation about z: quat_xyzw = [0, 0, sin(45), cos(45)].
    s = math.sin(math.radians(45))
    c = math.cos(math.radians(45))
    identity = [0.0, 0.0, 0.0, 1.0]
    rot_z_90 = [0.0, 0.0, s, c]
    assert orientation_error_deg(identity, rot_z_90) == pytest.approx(90.0, abs=1e-4)


def test_orientation_error_handles_unnormalized_input():
    # Scaling a quaternion must not change the represented orientation.
    identity = [0.0, 0.0, 0.0, 2.0]
    same = [0.0, 0.0, 0.0, 5.0]
    assert orientation_error_deg(identity, same) == pytest.approx(0.0, abs=1e-6)
