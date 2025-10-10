import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from ur_asu.custom_libraries.motion_utils import MotionExecutor
from ur_asu.custom_libraries.pusher_utils import PusherHandler
from ur_asu.custom_libraries.gripper_utils import GripperHandler, GRIPPER_TABLE

class MixedCartesianController(Node):
    def __init__(self):
        super().__init__('mixed_cartesian_controller')

        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # Controller Names
        self.position_controller = 'scaled_joint_trajectory_controller'
        self.velocity_controller = 'forward_velocity_controller'
        self.force_controller = 'force_mode_controller'
        self.passthrough_controller = 'passthrough_trajectory_controller'

        # Aux. Motion Handler
        self.pusher = PusherHandler(self)
        self.gripper = GripperHandler(self, vertical_offset = 0.005)
        self.motion = MotionExecutor(
            self, self.joint_names,
            self.position_controller,
            self.velocity_controller,
            self.force_controller,
            self.passthrough_controller,
            self.gripper,
            pusher=self.pusher
            )


def main(args=None):
    rclpy.init(args=args)
    node = MixedCartesianController()

    # Step 1: Go to a point
    node.motion.send_cartesian_position([0.1, -0.5, 0.247], [0, 180, 0], seconds=4)

    # Step 2: Cartesian drift -Y for 5s
    v_cart = np.array([0.0, -0.02, 0.0, 0.0, 0.0, 0.0])
    node.motion.send_cartesian_velocity(v_cart, duration=5.0)

    # Step 3: Cartesian drift +Y for 5s
    v_cart = np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0])
    node.motion.send_cartesian_velocity(v_cart, duration=5.0)

    # Step 4: Apply downward force for 3s
    selection_vector = [False, False, True, False, False, False]  # Only z-axis compliant
    wrench = [0.0, 0.0, -10.0, 0.0, 0.0, 0.0]  # Push downward

    node.motion.send_cartesian_force(wrench, 3.0, selection_vector=selection_vector)

    node.get_logger().info("Mixed Cartesian + Force motion complete.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
