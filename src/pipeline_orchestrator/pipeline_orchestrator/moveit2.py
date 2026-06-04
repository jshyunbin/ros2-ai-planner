from __future__ import annotations

import copy
import math
import threading
from typing import Any, Optional, Sequence

from moveit_msgs.srv import GetPositionIK

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from shape_msgs.msg import SolidPrimitive

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
    RobotState,
    JointConstraint,
    RobotState
)


class MoveIt2:
    """
    MoveIt2 planner module for pipeline_orchestrator.

    Input:
      grasp_pose:
        - geometry_msgs/PoseStamped
        - geometry_msgs/Pose
        - dict with x,y,z,qx,qy,qz,qw
        - list/tuple [x, y, z, qx, qy, qz, qw]

      joint_states:
        - sensor_msgs/JointState from /joint_states
        - if None, uses default joint positions from parameters

    Output:
      trajectory_msgs/JointTrajectory or None
    """

    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()

        self._group_name = self._declare_and_get("moveit2.group_name", "ur5_arm")
        self._end_effector_link = self._declare_and_get("moveit2.end_effector_link", "tool0")
        self._planning_frame = self._declare_and_get("moveit2.planning_frame", "world")
        self._move_action_name = self._declare_and_get("moveit2.move_action_name", "/move_action")

        self._pipeline_id = self._declare_and_get("moveit2.pipeline_id", "")
        self._planner_id = self._declare_and_get("moveit2.planner_id", "")

        self._allowed_planning_time = float(
            self._declare_and_get("moveit2.allowed_planning_time", 5.0)
        )
        self._num_planning_attempts = int(
            self._declare_and_get("moveit2.num_planning_attempts", 5)
        )
        self._velocity_scaling = float(
            self._declare_and_get("moveit2.velocity_scaling", 0.25)
        )
        self._acceleration_scaling = float(
            self._declare_and_get("moveit2.acceleration_scaling", 0.25)
        )

        self._position_tolerance = float(
            self._declare_and_get("moveit2.position_tolerance", 0.01)
        )
        self._orientation_tolerance = float(
            self._declare_and_get("moveit2.orientation_tolerance", 0.05)
        )

        self._server_timeout_sec = float(
            self._declare_and_get("moveit2.server_timeout_sec", 10.0)
        )
        self._result_timeout_sec = float(
            self._declare_and_get("moveit2.result_timeout_sec", 20.0)
        )

        self._arm_joint_names = list(
            self._declare_and_get(
                "moveit2.arm_joint_names",
                [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
            )
        )

        self._default_joint_positions = list(
            self._declare_and_get(
                "moveit2.default_joint_positions",
                [
                    0.0,
                    -1.5708,
                    1.0,
                    -1.0472,
                    -1.5708,
                    0.0,
                ],
            )
        )

        self._move_group_client = ActionClient(
            self._node,
            MoveGroup,
            self._move_action_name,
        )

        self._logger.info(
            "MoveIt2 planner initialized: "
            f"group={self._group_name}, "
            f"ee_link={self._end_effector_link}, "
            f"frame={self._planning_frame}, "
            f"action={self._move_action_name}"
        )

        self._ik_client = self._node.create_client(
            GetPositionIK,
            "/compute_ik",
        )

        self._arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

    def _nearest_equivalent_angle(self, target: float, current: float) -> float:
        """
        Choose an equivalent revolute-joint angle closest to current.

        Example:
          target = 6.10 rad, current = -0.18 rad
          returns approximately -0.18 rad instead of rotating one full turn.
        """
        while target - current > math.pi:
            target -= 2.0 * math.pi

        while target - current < -math.pi:
            target += 2.0 * math.pi

        return target

    def _solve_ik_near_current(self, target_pose, joint_state):
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self._node.get_logger().error("/compute_ik service is not available.")
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name = self._group_name
        req.ik_request.ik_link_name = self._end_effector_link
        req.ik_request.pose_stamped = target_pose
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 2
        req.ik_request.timeout.nanosec = 0

        seed_state = RobotState()
        seed_state.joint_state = joint_state
        req.ik_request.robot_state = seed_state

        future = self._ik_client.call_async(req)

        rclpy.spin_until_future_complete(
            self._node,
            future,
            timeout_sec=5.0,
        )

        if not future.done() or future.result() is None:
            self._node.get_logger().error("IK request timed out or failed.")
            return None

        res = future.result()

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().error(
                f"IK failed: error_code={res.error_code.val}"
            )
            return None

        solution = res.solution.joint_state

        current_map = {
            name: pos
            for name, pos in zip(joint_state.name, joint_state.position)
        }

        solution_map = {
            name: pos
            for name, pos in zip(solution.name, solution.position)
        }

        goal_positions = {}

        for joint_name in self._arm_joint_names:
            if joint_name not in solution_map:
                self._node.get_logger().error(
                    f"IK solution does not contain joint: {joint_name}"
                )
                return None

            raw_goal = float(solution_map[joint_name])
            current = float(current_map.get(joint_name, raw_goal))

            goal_positions[joint_name] = self._nearest_equivalent_angle(
                raw_goal,
                current,
            )

        return goal_positions

    def _build_joint_goal_constraints(self, goal_positions):
        constraints = Constraints()
        constraints.name = "ik_joint_goal"

        for joint_name, position in goal_positions.items():
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(position)
            jc.tolerance_above = 0.005
            jc.tolerance_below = 0.005
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        return constraints

    def _declare_and_get(self, name: str, default_value: Any) -> Any:
        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, default_value)
        return self._node.get_parameter(name).value

    def plan_trajectory(self, target_pose, joint_state):
        goal_positions = self._solve_ik_near_current(
            target_pose,
            joint_state,
        )

        if goal_positions is None:
            self._node.get_logger().error("Failed to compute nearby IK goal.")
            return None

        self._node.get_logger().info("Nearby IK joint goal:")
        for name, value in goal_positions.items():
            self._node.get_logger().info(f"  {name}: {value:.4f}")

        goal_msg = MoveGroup.Goal()

        req = self._build_motion_plan_request(
            target_pose,
            joint_state,
        )

        # 중요: 기존 pose constraint 제거하고 joint constraint로 대체
        req.goal_constraints = [
            self._build_joint_goal_constraints(goal_positions)
        ]

        goal_msg.request = req
        goal_msg.planning_options = self._build_planning_options()

        if not self._move_group_client.wait_for_server(timeout_sec=10.0):
            self._node.get_logger().error("MoveGroup action server is not available.")
            return None

        send_goal_future = self._move_group_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(
            self._node,
            send_goal_future,
            timeout_sec=10.0,
        )

        if not send_goal_future.done():
            self._node.get_logger().error("Timed out while sending MoveIt2 goal.")
            return None

        goal_handle = send_goal_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error("MoveIt2 goal rejected.")
            return None

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self._node,
            result_future,
            timeout_sec=60.0,
        )

        if not result_future.done():
            self._node.get_logger().error("MoveIt2 planning result timed out.")
            return None

        result = result_future.result().result

        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().error(
                f"MoveIt2 planning failed: error_code={result.error_code.val}"
            )
            return None

        trajectory = result.planned_trajectory.joint_trajectory

        if len(trajectory.points) == 0:
            self._node.get_logger().error("MoveIt2 returned empty trajectory.")
            return None

        self._node.get_logger().info(
            f"MoveIt2 planning succeeded: {len(trajectory.points)} points."
        )

        return trajectory

    def _build_motion_plan_request(
        self,
        target_pose: PoseStamped,
        joint_states: Optional[JointState],
    ) -> MotionPlanRequest:
        req = MotionPlanRequest()

        req.group_name = self._group_name
        if self._pipeline_id:
            req.pipeline_id = self._pipeline_id

        if self._planner_id:
            req.planner_id = self._planner_id

        req.num_planning_attempts = self._num_planning_attempts
        req.allowed_planning_time = self._allowed_planning_time
        req.max_velocity_scaling_factor = self._velocity_scaling
        req.max_acceleration_scaling_factor = self._acceleration_scaling

        req.start_state = self._build_start_state(joint_states)
        req.goal_constraints = [
            self._build_pose_goal_constraints(target_pose)
        ]
        req.start_state.joint_state = joint_states
        req.start_state.is_diff = False


        req.workspace_parameters.header.frame_id = self._planning_frame
        req.workspace_parameters.min_corner.x = -2.0
        req.workspace_parameters.min_corner.y = -2.0
        req.workspace_parameters.min_corner.z = -1.0
        req.workspace_parameters.max_corner.x = 2.0
        req.workspace_parameters.max_corner.y = 2.0
        req.workspace_parameters.max_corner.z = 2.0

        return req

    def _build_planning_options(self) -> PlanningOptions:
        options = PlanningOptions()

        # Plan only. Do not execute inside MoveIt.
        options.plan_only = True
        options.look_around = False
        options.replan = False

        options.planning_scene_diff.is_diff = True
        options.planning_scene_diff.robot_state.is_diff = True

        return options

    def _build_start_state(
        self,
        joint_states: Optional[JointState],
    ) -> RobotState:
        state = RobotState()
        state.is_diff = True

        if joint_states is None or not joint_states.name:
            self._logger.warn(
                "No valid JointState supplied. Using default joint positions."
            )

            js = JointState()
            js.name = list(self._arm_joint_names)
            js.position = [float(v) for v in self._default_joint_positions]
            state.joint_state = js
            return state

        source_index = {
            name: idx
            for idx, name in enumerate(joint_states.name)
        }

        filtered = JointState()
        filtered.header = joint_states.header

        missing = []

        for joint_name in self._arm_joint_names:
            if joint_name not in source_index:
                missing.append(joint_name)
                continue

            idx = source_index[joint_name]
            filtered.name.append(joint_name)

            if idx < len(joint_states.position):
                filtered.position.append(float(joint_states.position[idx]))

        if missing:
            self._logger.warn(
                f"Missing configured arm joints in JointState: {missing}"
            )

        if not filtered.name:
            self._logger.warn(
                "No configured arm joints matched JointState. "
                "Using default joint positions."
            )
            filtered.name = list(self._arm_joint_names)
            filtered.position = [float(v) for v in self._default_joint_positions]

        state.joint_state = filtered
        return state

    def _build_pose_goal_constraints(
        self,
        target_pose: PoseStamped,
    ) -> Constraints:
        constraints = Constraints()
        constraints.name = "target_ee_pose"

        constraints.position_constraints.append(
            self._build_position_constraint(target_pose)
        )
        constraints.orientation_constraints.append(
            self._build_orientation_constraint(target_pose)
        )

        return constraints

    def _build_position_constraint(
        self,
        target_pose: PoseStamped,
    ) -> PositionConstraint:
        pc = PositionConstraint()

        pc.header.frame_id = target_pose.header.frame_id
        pc.header.stamp = self._node.get_clock().now().to_msg()
        pc.link_name = self._end_effector_link
        pc.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self._position_tolerance]

        sphere_pose = Pose()
        sphere_pose.position.x = target_pose.pose.position.x
        sphere_pose.position.y = target_pose.pose.position.y
        sphere_pose.position.z = target_pose.pose.position.z
        sphere_pose.orientation.w = 1.0

        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(sphere_pose)

        pc.constraint_region = region
        return pc

    def _build_orientation_constraint(
        self,
        target_pose: PoseStamped,
    ) -> OrientationConstraint:
        oc = OrientationConstraint()

        oc.header.frame_id = target_pose.header.frame_id
        oc.header.stamp = self._node.get_clock().now().to_msg()
        oc.link_name = self._end_effector_link
        oc.orientation = target_pose.pose.orientation

        oc.absolute_x_axis_tolerance = self._orientation_tolerance
        oc.absolute_y_axis_tolerance = self._orientation_tolerance
        oc.absolute_z_axis_tolerance = self._orientation_tolerance
        oc.parameterization = OrientationConstraint.ROTATION_VECTOR
        oc.weight = 1.0

        return oc

    def _to_pose_stamped(self, grasp_pose: Any) -> Optional[PoseStamped]:
        if grasp_pose is None:
            return None

        if isinstance(grasp_pose, PoseStamped):
            pose_stamped = copy.deepcopy(grasp_pose)

        elif isinstance(grasp_pose, Pose):
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self._planning_frame
            pose_stamped.pose = copy.deepcopy(grasp_pose)

        elif isinstance(grasp_pose, dict):
            pose_stamped = self._dict_to_pose_stamped(grasp_pose)

        elif isinstance(grasp_pose, Sequence) and len(grasp_pose) == 7:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self._planning_frame

            pose_stamped.pose.position.x = float(grasp_pose[0])
            pose_stamped.pose.position.y = float(grasp_pose[1])
            pose_stamped.pose.position.z = float(grasp_pose[2])

            pose_stamped.pose.orientation.x = float(grasp_pose[3])
            pose_stamped.pose.orientation.y = float(grasp_pose[4])
            pose_stamped.pose.orientation.z = float(grasp_pose[5])
            pose_stamped.pose.orientation.w = float(grasp_pose[6])

        else:
            self._logger.error(
                "Unsupported grasp_pose type. Expected PoseStamped, Pose, dict, "
                "or [x, y, z, qx, qy, qz, qw]."
            )
            return None

        if not pose_stamped.header.frame_id:
            pose_stamped.header.frame_id = self._planning_frame

        pose_stamped.header.stamp = self._node.get_clock().now().to_msg()
        self._normalize_quaternion_in_place(pose_stamped.pose)

        return pose_stamped

    def _dict_to_pose_stamped(self, data: dict) -> PoseStamped:
        pose_stamped = PoseStamped()

        pose_data = data.get("pose", data)
        position = pose_data.get("position", pose_data)
        orientation = pose_data.get("orientation", pose_data)

        pose_stamped.header.frame_id = (
            data.get("frame_id")
            or data.get("header", {}).get("frame_id", "")
            or self._planning_frame
        )

        pose_stamped.pose.position.x = float(
            position.get("x", pose_data.get("x", 0.0))
        )
        pose_stamped.pose.position.y = float(
            position.get("y", pose_data.get("y", 0.0))
        )
        pose_stamped.pose.position.z = float(
            position.get("z", pose_data.get("z", 0.0))
        )

        pose_stamped.pose.orientation.x = float(
            orientation.get("x", pose_data.get("qx", 0.0))
        )
        pose_stamped.pose.orientation.y = float(
            orientation.get("y", pose_data.get("qy", 0.0))
        )
        pose_stamped.pose.orientation.z = float(
            orientation.get("z", pose_data.get("qz", 0.0))
        )
        pose_stamped.pose.orientation.w = float(
            orientation.get("w", pose_data.get("qw", 1.0))
        )

        return pose_stamped

    def _normalize_quaternion_in_place(self, pose: Pose) -> None:
        q = pose.orientation

        norm = math.sqrt(
            q.x * q.x
            + q.y * q.y
            + q.z * q.z
            + q.w * q.w
        )

        if norm < 1e-9:
            self._logger.warn(
                "Target quaternion is near zero. Using identity orientation."
            )
            q.x = 0.0
            q.y = 0.0
            q.z = 0.0
            q.w = 1.0
            return

        q.x /= norm
        q.y /= norm
        q.z /= norm
        q.w /= norm