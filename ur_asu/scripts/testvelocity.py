import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
from ur_asu.custom_libraries.ik_solver import compute_jacobian  # your file!

class CartesianVelocityController(Node):
    def __init__(self):
        super().__init__('cartesian_velocity_controller')
        self.get_logger().info("Starting CartesianVelocityController with scripted segments.")

        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.n_joints = len(self.joint_names)

        # Sequence of (velocity vector, duration) pairs
        vel = 0.02
        duration = 5.0
        self.velocity_sequence = [
            (np.array([ vel,  0.0, 0.0, 0.0, 0.0, 0.0]), duration),  # +X
            (np.array([ 0.0,  vel, 0.0, 0.0, 0.0, 0.0]), duration),  # +Y
            (np.array([-vel,  0.0, 0.0, 0.0, 0.0, 0.0]), duration),  # -X
            (np.array([ 0.0, -vel, 0.0, 0.0, 0.0, 0.0]), duration),  # -Y
            (np.zeros(6), 2.0),                                  # stop 2s
        ]
        self.segment_index = 0
        self.segment_start_time = None
        self.segment_index = 0
        self.v_cartesian = np.zeros(6)  # will be updated later
        # self.v_cartesian = self.velocity_sequence[0][0]  # initialize

        self.joint_vel_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        self.joint_positions = None
        self.joint_positions_map = {}

        self.timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions_map[name] = pos

        if all(name in self.joint_positions_map for name in self.joint_names):
            self.joint_positions = np.array([self.joint_positions_map[name] for name in self.joint_names])

    def control_loop(self):
        if self.joint_positions is None:
            self.get_logger().warn("Waiting for joint_states...")
            return

        # Set start time and initial velocity when data becomes ready
        if self.segment_start_time is None:
            self.segment_start_time = self.get_clock().now()
            self.v_cartesian = self.velocity_sequence[0][0]
            self.get_logger().info("▶ Starting first segment")
            return  # Skip sending velocity on this cycle

        now = self.get_clock().now()
        elapsed = (now - self.segment_start_time).nanoseconds / 1e9
        _, duration = self.velocity_sequence[self.segment_index]

        if elapsed >= duration:
            self.segment_index += 1
            if self.segment_index >= len(self.velocity_sequence):
                self.get_logger().info("✔ All segments complete. Stopping.")
                self.v_cartesian = np.zeros(6)
                self.timer.cancel()
                return
            else:
                self.segment_start_time = now
                self.v_cartesian = self.velocity_sequence[self.segment_index][0]
                self.get_logger().info(f"▶ Switching to segment {self.segment_index + 1}/{len(self.velocity_sequence)}")

        try:
            J = compute_jacobian(self.joint_positions)
            q_dot = np.linalg.pinv(J) @ self.v_cartesian
        except Exception as e:
            self.get_logger().error(f"Jacobian computation failed: {e}")
            return

        msg = Float64MultiArray()
        msg.data = q_dot.tolist()
        self.joint_vel_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CartesianVelocityController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down via KeyboardInterrupt.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
