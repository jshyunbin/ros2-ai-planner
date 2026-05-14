from rclpy.node import Node


class MoveIt2:
    """MoveIt2 motion planning module — fallback when cuRobo fails.

    Unlike the other modules, MoveIt2 is ROS2-native: it communicates with
    the move_group node via action/service clients and therefore requires a
    live ROS2 node handle rather than just a logger.

    Implementation options (choose one when implementing):
      A) moveit_py  — official Python bindings shipped with ros-humble-moveit
      B) pymoveit2  — third-party wrapper around move_group actions/services
                      (pip install pymoveit2)
    """

    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()
        # TODO: initialize MoveIt2 client using moveit_py or pymoveit2

    def plan_trajectory(self, grasp_pose, joint_states):
        """Plan trajectory to grasp_pose using MoveIt2. Returns trajectory or None."""
        # TODO: call move_group plan via MoveIt2 client
        self._logger.warn('MoveIt2.plan_trajectory not yet implemented.')
        return None
