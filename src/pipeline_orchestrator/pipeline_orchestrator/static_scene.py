#!/usr/bin/env python3

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive


def yaw_rotate_xy(x: float, y: float, yaw: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


class StaticScenePublisher(Node):
    """
    Add fixed manip_challenge set2 objects to the MoveIt2 planning scene.

    Assumption:
      - frame_id is "world"
      - MoveIt2 robot_description already has world -> base_link z=0.6 offset
      - Gazebo/controller are running on host
      - move_group is running in Docker
    """

    def __init__(self):
        super().__init__("add_static_scene_to_moveit")

        self.declare_parameter("frame_id", "world")
        self.declare_parameter("apply_service", "/apply_planning_scene")
        self.declare_parameter("padding", 0.0)

        self.declare_parameter("include_table", True)
        self.declare_parameter("table_center_x", 0.7)
        self.declare_parameter("table_center_y", 0.0)
        self.declare_parameter("table_center_z", 0.48)
        self.declare_parameter("table_size_x", 1.20)
        self.declare_parameter("table_size_y", 0.80)
        self.declare_parameter("table_thickness", 0.06)

        self.declare_parameter("include_storage_bases", True)
        self.declare_parameter("include_center_ur5_base", False)
        self.declare_parameter("include_workspace_basket_bottom", False)

        self.frame_id = self.get_parameter("frame_id").value
        self.apply_service = self.get_parameter("apply_service").value
        self.padding = float(self.get_parameter("padding").value)

        self.client = self.create_client(ApplyPlanningScene, self.apply_service)
        self.done = False

        self.get_logger().info(f"Waiting for {self.apply_service} service...")

        if not self.client.wait_for_service(timeout_sec=20.0):
            self.get_logger().error(f"{self.apply_service} is not available.")
            self.done = True
            return

        self.apply_scene()

    def make_box(
        self,
        object_id: str,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        yaw: float = 0.0,
    ) -> CollisionObject:
        obj = CollisionObject()
        obj.header.frame_id = self.frame_id
        obj.id = object_id

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [
            float(size[0] + 2.0 * self.padding),
            float(size[1] + 2.0 * self.padding),
            float(size[2] + 2.0 * self.padding),
        ]

        pose = Pose()
        pose.position.x = float(center[0])
        pose.position.y = float(center[1])
        pose.position.z = float(center[2])

        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        return obj

    def add_model_box(
        self,
        objects: List[CollisionObject],
        object_id: str,
        model_xyz: Tuple[float, float, float],
        model_yaw: float,
        local_xyz: Tuple[float, float, float],
        size: Tuple[float, float, float],
        local_yaw: float = 0.0,
    ):
        lx, ly, lz = local_xyz
        rx, ry = yaw_rotate_xy(lx, ly, model_yaw)

        center = (
            model_xyz[0] + rx,
            model_xyz[1] + ry,
            model_xyz[2] + lz,
        )

        objects.append(
            self.make_box(
                object_id=object_id,
                center=center,
                size=size,
                yaw=model_yaw + local_yaw,
            )
        )

    def add_table(self, objects: List[CollisionObject]):
        if not bool(self.get_parameter("include_table").value):
            return

        objects.append(
            self.make_box(
                object_id="cafe_table_top",
                center=(
                    float(self.get_parameter("table_center_x").value),
                    float(self.get_parameter("table_center_y").value),
                    float(self.get_parameter("table_center_z").value),
                ),
                size=(
                    float(self.get_parameter("table_size_x").value),
                    float(self.get_parameter("table_size_y").value),
                    float(self.get_parameter("table_thickness").value),
                ),
                yaw=0.0,
            )
        )

    def add_ur5_base(
        self,
        objects: List[CollisionObject],
        prefix: str,
        model_xyz: Tuple[float, float, float],
        include_top: bool = True,
    ):
        """
        Approximate ur5_base model.

        top plate:
          local center = (0, 0, 0.58)
          size         = (0.5, 0.5, 0.02)

        four legs:
          local centers = (+/-0.22, +/-0.22, 0.285)
          size          = (0.04, 0.04, 0.57)
        """

        yaw = 0.0

        if include_top:
            self.add_model_box(
                objects,
                f"{prefix}_top",
                model_xyz,
                yaw,
                local_xyz=(0.0, 0.0, 0.58),
                size=(0.5, 0.5, 0.02),
            )

        leg_positions = [
            (0.22, 0.22, 0.285),
            (-0.22, 0.22, 0.285),
            (0.22, -0.22, 0.285),
            (-0.22, -0.22, 0.285),
        ]

        for i, local_xyz in enumerate(leg_positions):
            self.add_model_box(
                objects,
                f"{prefix}_leg_{i}",
                model_xyz,
                yaw,
                local_xyz=local_xyz,
                size=(0.04, 0.04, 0.57),
            )

    def add_storage_basket(
        self,
        objects: List[CollisionObject],
        prefix: str,
        model_xyz: Tuple[float, float, float],
    ):
        yaw = 0.0

        self.add_model_box(
            objects,
            f"{prefix}_bottom",
            model_xyz,
            yaw,
            local_xyz=(0.0, 0.0, 0.0025),
            size=(0.35, 0.45, 0.005),
        )

        self.add_model_box(
            objects,
            f"{prefix}_wall_pos_y",
            model_xyz,
            yaw,
            local_xyz=(0.0, 0.225, 0.0275),
            size=(0.35, 0.005, 0.05),
        )

        self.add_model_box(
            objects,
            f"{prefix}_wall_neg_y",
            model_xyz,
            yaw,
            local_xyz=(0.0, -0.225, 0.0275),
            size=(0.35, 0.005, 0.05),
        )

        self.add_model_box(
            objects,
            f"{prefix}_wall_pos_x",
            model_xyz,
            yaw,
            local_xyz=(0.175, 0.0, 0.0275),
            size=(0.005, 0.45, 0.05),
        )

        self.add_model_box(
            objects,
            f"{prefix}_wall_neg_x",
            model_xyz,
            yaw,
            local_xyz=(-0.175, 0.0, 0.0275),
            size=(0.005, 0.45, 0.05),
        )

    def add_workspace_basket(
        self,
        objects: List[CollisionObject],
        model_xyz: Tuple[float, float, float],
    ):
        prefix = "workspace_basket"
        yaw = 0.0

        include_bottom = bool(
            self.get_parameter("include_workspace_basket_bottom").value
        )

        if include_bottom:
            self.add_model_box(
                objects,
                f"{prefix}_bottom",
                model_xyz,
                yaw,
                local_xyz=(0.0, 0.0, 0.005),
                size=(0.5, 0.9, 0.01),
            )

        self.add_model_box(
            objects,
            f"{prefix}_wall_pos_y",
            model_xyz,
            yaw,
            local_xyz=(0.0, 0.45, 0.0275),
            size=(0.5, 0.005, 0.05),
        )

        self.add_model_box(
            objects,
            f"{prefix}_wall_neg_y",
            model_xyz,
            yaw,
            local_xyz=(0.0, -0.45, 0.0275),
            size=(0.5, 0.005, 0.05),
        )

        self.add_model_box(
            objects,
            f"{prefix}_wall_pos_x",
            model_xyz,
            yaw,
            local_xyz=(0.25, 0.0, 0.0275),
            size=(0.005, 0.9, 0.05),
        )

        self.add_model_box(
            objects,
            f"{prefix}_wall_neg_x",
            model_xyz,
            yaw,
            local_xyz=(-0.25, 0.0, 0.0275),
            size=(0.005, 0.9, 0.05),
        )

    def add_bookshelf(
        self,
        objects: List[CollisionObject],
        model_xyz: Tuple[float, float, float],
        model_yaw: float,
    ):
        prefix = "bookshelf"

        for i, z in enumerate([0.01, 0.23, 0.45]):
            self.add_model_box(
                objects,
                f"{prefix}_shelf_{i}",
                model_xyz,
                model_yaw,
                local_xyz=(0.0, 0.0, z),
                size=(0.3, 0.27, 0.02),
            )

        self.add_model_box(
            objects,
            f"{prefix}_side_wall",
            model_xyz,
            model_yaw,
            local_xyz=(0.0, 0.125, 0.23),
            size=(0.3, 0.02, 0.46),
        )

    def apply_scene(self):
        objects: List[CollisionObject] = []

        self.add_table(objects)

        # Tables under storage basket A/B
        if bool(self.get_parameter("include_storage_bases").value):
            self.add_ur5_base(
                objects,
                prefix="ur5_base_storage_a",
                model_xyz=(0.0, 0.55, 0.0),
                include_top=True,
            )

            self.add_ur5_base(
                objects,
                prefix="ur5_base_storage_b",
                model_xyz=(0.0, -0.55, 0.0),
                include_top=True,
            )

        # Center base is directly under the robot.
        # Do not include its top plate by default because it can cause
        # start-state collision with the robot base.
        if bool(self.get_parameter("include_center_ur5_base").value):
            self.add_ur5_base(
                objects,
                prefix="ur5_base_center",
                model_xyz=(0.0, 0.0, 0.0),
                include_top=False,
            )

        self.add_storage_basket(
            objects,
            prefix="storage_a_basket",
            model_xyz=(0.0, 0.55, 0.6),
        )

        self.add_storage_basket(
            objects,
            prefix="storage_b_basket",
            model_xyz=(0.0, -0.55, 0.6),
        )

        self.add_workspace_basket(
            objects,
            model_xyz=(0.55, 0.0, 0.5),
        )

        self.add_bookshelf(
            objects,
            model_xyz=(0.95, -0.3, 0.5),
            model_yaw=-1.57,
        )

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objects

        request = ApplyPlanningScene.Request()
        request.scene = scene

        self.get_logger().info(
            f"Applying {len(objects)} static collision objects "
            f"to MoveIt2 planning scene in frame '{self.frame_id}'"
        )

        future = self.client.call_async(request)
        future.add_done_callback(self._on_done)

    def _on_done(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"ApplyPlanningScene call failed: {exc}")
            self.done = True
            return

        if response.success:
            self.get_logger().info("Static scene added to MoveIt2 planning scene.")
        else:
            self.get_logger().error("ApplyPlanningScene returned success=False.")

        self.done = True


def main(args=None):
    rclpy.init(args=args)

    node = StaticScenePublisher()

    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()