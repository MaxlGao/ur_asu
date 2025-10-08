import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Wrench, Twist

from controller_manager_msgs.srv import SwitchController, ListControllers, LoadController
from ur_msgs.srv import SetForceMode
from std_srvs.srv import Trigger

from ur_asu.custom_libraries.ik_solver import compute_ik, compute_jacobian


class MixedCartesianController(Node):
    def __init__(self):
        super().__init__('mixed_cartesian_controller')

        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # Controllers
        self.position_controller = 'scaled_joint_trajectory_controller'
        self.velocity_controller = 'forward_velocity_controller'
        self.force_controller = 'force_mode_controller'
        self.passthrough_controller = 'passthrough_trajectory_controller'

        # Action client for position control
        traj_action_name = f'/{self.position_controller}/follow_joint_trajectory'
        self.traj_client = ActionClient(self, FollowJointTrajectory, traj_action_name)

        # Velocity publisher
        self.joint_vel_pub = self.create_publisher(
            Float64MultiArray,
            f'/{self.velocity_controller}/commands',
            10
        )
        
        # Service clients
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')

        # Force mode clients
        self.start_force_client = self.create_client(SetForceMode, '/force_mode_controller/start_force_mode')
        self.stop_force_client = self.create_client(Trigger, '/force_mode_controller/stop_force_mode')

        # Joint state subscription
        self.joint_positions = None
        self.joint_positions_map = {}
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        
        # Pre-start controllers automatically
        self.prestart_controllers()

    def list_controllers(self):
        if not self.list_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("ListControllers service not available.")
            return []
        future = self.list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        return result.controller if result else []

    def prestart_controllers(self):
        """Ensure all required controllers are loaded and at least one is active."""
        required = [
            self.position_controller,
            self.velocity_controller,
            self.force_controller,
            self.passthrough_controller
        ]
        controllers = {c.name: c.state for c in self.list_controllers()}
        self.get_logger().info(f"Detected controllers: {controllers}")

        for ctrl in required:
            if ctrl not in controllers:
                # Try to load missing controller
                self.get_logger().warn(f"{ctrl} not loaded. Attempting to load...")
                if not self.load_client.wait_for_service(timeout_sec=2.0):
                    self.get_logger().error("LoadController service not available.")
                    continue
                req = LoadController.Request()
                req.name = ctrl
                future = self.load_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                res = future.result()
                if not res or not res.ok:
                    self.get_logger().error(f"Failed to load {ctrl}")
                    continue

        # Start the default position controller to ensure robot can move
        self.switch_to_controller([self.position_controller], [])

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions_map[name] = pos
        if all(name in self.joint_positions_map for name in self.joint_names):
            self.joint_positions = np.array([self.joint_positions_map[name] for name in self.joint_names])

    def get_active_controllers(self):
        active = {ctrl.name for ctrl in self.list_controllers() if ctrl.state == 'active'}
        return active

    def switch_to_controller(self, start_list, stop_list):
        """Smartly switch active controllers without redundant calls."""
        active_controllers = self.get_active_controllers()

        # Filter out already active/inactive controllers
        to_start = [c for c in start_list if c not in active_controllers]
        to_stop = [c for c in stop_list if c in active_controllers]

        if not to_start and not to_stop:
            self.get_logger().info(
                f"No controller switch needed. Active: {sorted(active_controllers)}"
            )
            return True

        self.get_logger().info(
            f"Switching controllers -> start: {to_start or '[]'}, stop: {to_stop or '[]'}"
        )

        if not self.switch_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Controller switch service not available.")
            return False

        req = SwitchController.Request()
        req.activate_controllers = to_start
        req.deactivate_controllers = to_stop
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        req.timeout = Duration(sec=5)

        future = self.switch_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if not result or not result.ok:
            self.get_logger().error(f"Controller switch failed. Tried to start={to_start}, stop={to_stop}")
            return False

        # Post verification
        active_controllers_after = self.get_active_controllers()
        self.get_logger().info(
            f"Controller switch success. Now active: {sorted(active_controllers_after)}"
        )
        return True

    def send_cartesian_position_goal(self, position, rpy_deg, seconds=3.0):
        self.switch_to_controller([self.position_controller], [self.velocity_controller, self.force_controller, self.passthrough_controller])
        joint_positions = compute_ik(position, rpy_deg)
        if joint_positions is None:
            self.get_logger().error("IK failed for position goal.")
            return False

        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.time_from_start = Duration(sec=int(seconds))
        traj.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.traj_client.wait_for_server()
        self.get_logger().info(f"Sending Cartesian position goal: {position}, rpy={rpy_deg}")
        future = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Position goal rejected by controller.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Position goal completed.")
        return True

    def send_cartesian_velocity(self, v_cartesian, duration):
        self.switch_to_controller([self.velocity_controller], [self.position_controller, self.force_controller, self.passthrough_controller])
        start_time = time.time()
        self.get_logger().info(f"Starting velocity phase: {v_cartesian} for {duration}s")
        while rclpy.ok() and (time.time() - start_time) < duration:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.joint_positions is None:
                self.get_logger().warn("Waiting for joint state...")
                continue
            try:
                J = compute_jacobian(self.joint_positions)
                q_dot = np.linalg.pinv(J) @ v_cartesian
            except Exception as e:
                self.get_logger().error(f"Jacobian computation failed: {e}")
                continue

            msg = Float64MultiArray()
            msg.data = q_dot.tolist()
            self.joint_vel_pub.publish(msg)
        self.stop_velocity()

    def stop_velocity(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * len(self.joint_names)
        self.joint_vel_pub.publish(msg)
        self.get_logger().info("Stopped velocity controller.")

    def start_force_mode(self, task_frame_pose, selection_vector, wrench, vel_limits, pos_limits, damping=0.025, gain=0.5):
        """Activate UR Force Mode through service."""
        self.switch_to_controller([self.force_controller, self.passthrough_controller],
                                  [self.position_controller, self.velocity_controller])

        if not self.start_force_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Force mode start service not available.")
            return False

        req = SetForceMode.Request()
        req.task_frame = task_frame_pose
        req.selection_vector_x = selection_vector[0]
        req.selection_vector_y = selection_vector[1]
        req.selection_vector_z = selection_vector[2]
        req.selection_vector_rx = selection_vector[3]
        req.selection_vector_ry = selection_vector[4]
        req.selection_vector_rz = selection_vector[5]

        req.wrench = Wrench()
        req.wrench.force.x, req.wrench.force.y, req.wrench.force.z = wrench[0:3]
        req.wrench.torque.x, req.wrench.torque.y, req.wrench.torque.z = wrench[3:6]

        req.type = 2  # Force frame not transformed
        req.speed_limits = Twist()
        req.speed_limits.linear.x, req.speed_limits.linear.y, req.speed_limits.linear.z = vel_limits[0:3]
        req.speed_limits.angular.x, req.speed_limits.angular.y, req.speed_limits.angular.z = vel_limits[3:6]

        req.deviation_limits = pos_limits # Keeping it the same; don't care too much
        req.damping_factor = damping
        req.gain_scaling = gain

        self.get_logger().info("Activating force mode...")
        future = self.start_force_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if not result or not result.success:
            self.get_logger().error(f"Failed to start force mode: {getattr(result, 'message', '')}")
            return False

        self.get_logger().info("Force mode activated.")
        return True

    def stop_force_mode(self):
        if not self.stop_force_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Force mode stop service not available.")
            return False

        future = self.stop_force_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if not result or not result.success:
            self.get_logger().error("Failed to stop force mode.")
            return False

        self.get_logger().info("Force mode stopped.")
        return True



def main(args=None):
    rclpy.init(args=args)
    node = MixedCartesianController()

    # Step 1: Go to a point
    node.send_cartesian_position_goal([0.1, -0.5, 0.147], [0, 180, 0], seconds=4)

    # Step 2: Cartesian drift -Y for 5s
    v_cart = np.array([0.0, -0.02, 0.0, 0.0, 0.0, 0.0])
    node.send_cartesian_velocity(v_cart, duration=5.0)

    # Step 3: Cartesian drift +Y for 5s
    v_cart = np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0])
    node.send_cartesian_velocity(v_cart, duration=5.0)

    # Step 4: Apply downward force for 3s
    task_frame = PoseStamped()
    task_frame.header.frame_id = "base"
    task_frame.pose.orientation.w = 1.0  # Identity rotation
    selection_vector = [False, False, True, False, False, False]  # Only z-axis compliant
    wrench = [0.0, 0.0, -10.0, 0.0, 0.0, 0.0]  # Push downward
    vel_limits = [0.25, 0.25, 0.25, 0.5, 0.5, 0.5] # Velocity limits
    pos_limits = [0.25, 0.25, 0.25, 0.5, 0.5, 0.5] # Position limits

    node.start_force_mode(task_frame, selection_vector, wrench, vel_limits, pos_limits)
    time.sleep(3.0)
    node.stop_force_mode()

    node.get_logger().info("Mixed Cartesian + Force motion complete.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
