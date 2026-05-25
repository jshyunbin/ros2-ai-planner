from sensor_msgs.msg import JointState


class CuRobo:
    """cuRobo motion planning module."""

    def __init__(self, logger):
        self._logger = logger
        # TODO: initialize cuRobo RobotConfig and MotionGenConfig

    def plan_trajectory(self, grasp_pose, joint_states: JointState, esdf=None):
        """Plan joint trajectory from current state to grasp pose.

        Args:
            grasp_pose: Target end-effector pose.
            joint_states: Current robot joint state.
            esdf: nvblox ESDF PointCloud2 from NvBlox.get_esdf(). When provided,
                  used to construct WorldNvbloxCollision for collision checking.
                  If None, plans without a dynamic collision world.

        Returns trajectory or None on failure.
        """
        # TODO: if esdf provided, construct WorldNvbloxCollision from esdf
        # TODO: initialize MotionGen with WorldNvbloxCollision or WorldPrimitiveCollision
        # TODO: call MotionGen.plan_single(start_state, goal_pose)
        self._logger.warn('CuRobo.plan_trajectory not yet implemented.')
        return None
