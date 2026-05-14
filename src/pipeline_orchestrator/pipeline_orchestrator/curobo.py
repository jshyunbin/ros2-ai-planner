from sensor_msgs.msg import JointState


class CuRobo:
    """cuRobo motion planning module."""

    def __init__(self, logger):
        self._logger = logger
        # TODO: initialize cuRobo WorldConfig, RobotConfig, MotionGenConfig

    def plan_trajectory(self, grasp_pose, joint_states: JointState):
        """Plan joint trajectory from current state to grasp pose. Returns trajectory or None."""
        # TODO: run cuRobo MotionGen
        self._logger.warn('CuRobo.plan_trajectory not yet implemented.')
        return None
