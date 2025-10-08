import numpy as np
import rclpy
import csv, os, datetime, threading, time
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, PointStamped, WrenchStamped
from scipy.spatial.transform import Rotation as R

from ur_asu.custom_libraries.gripperlibraries import *
from ur_asu.custom_libraries.controller_utils import ControllerManager
from ur_asu.custom_libraries.motion_utils import MotionExecutor
from ur_asu.custom_libraries.gripper_utils import GripperHandler
from ur_asu.custom_libraries.pusher_utils import PusherHandler
from ur_asu.custom_libraries.wrench_utils import WrenchHandler
from ur_asu.custom_libraries.generallibraries import canonicalize_euler, vector3_text

VERTICAL_OFFSET = 0.05

class MixedCartesianController(Node):
    def __init__(self):
        super().__init__('mixed_cartesian_controller')
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.position_controller = 'scaled_joint_trajectory_controller'
        self.velocity_controller = 'forward_velocity_controller'

        # Interfacing
        self.gripper_pub = self.create_publisher(String, "/gripper_command", 10)
        self.pusher_pub_l = self.create_publisher(PointStamped, "/gripper_pusher_left", 10)
        self.pusher_pub_r = self.create_publisher(PointStamped, "/gripper_pusher_right", 10)
        self.normed_wrench_pub = self.create_publisher(WrenchStamped, "/force_torque_sensor_smoothed_normed/wrench", 10)
        self.smoothed_wrench_pub = self.create_publisher(WrenchStamped, "/force_torque_sensor_smoothed/wrench", 10)
        self.baseline_wrench_pub = self.create_publisher(WrenchStamped, "/force_torque_sensor_baseline/wrench", 10)
        self.create_subscription(JointState, '/joint_states', 
                                 self.joint_state_callback, 10)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', 
                                 self.ee_pose_callback, 10)
        self.create_subscription(WrenchStamped, '/bota_driver_node/wrench',
                                 self.wrench_callback, 10)

        # Auxiliary Handlers
        self.controller_manager = ControllerManager(self)
        self.motion = MotionExecutor(self, self.joint_names,
                                     self.position_controller,
                                     self.velocity_controller,
                                     self.controller_manager)
        self.gripper = GripperHandler(self.gripper_pub, vertical_offset = 0.003)
        self.pusher = PusherHandler(self.pusher_pub_l, self.pusher_pub_r, self)
        self.wrench = WrenchHandler(self.smoothed_wrench_pub, self.baseline_wrench_pub, self.normed_wrench_pub, self)

        # Position States
        self.joint_positions = None
        self.ee_position = []
        self.ee_quat = []
        self.ee_euler = []
        
        # Force/Torque Sensing
        self.alpha = 0.01  # smoothing factor for force filter
        self.filtered_force = None # F/T smoothed by filter
        self.filtered_torque = None
        self.adjusted_force = None # smoothed F/T subtracted by offset 
        self.adjusted_torque = None
        self.force_offset = np.array([0.0, 0.0, 0.0]) # Mean F/T from static measure
        self.torque_offset = np.array([0.0, 0.0, 0.0])
        self.wrench_offset_initialized = False
        self.offset_init_duration = 3.0  # seconds
        self.offset_init_start_time = None
        self.offset_force_accumulator = []
        self.offset_torque_accumulator = []

        self.sensor_force = []
        self.sensor_torque = []

        # Logging pushes to CSV
        log_dir = os.path.expanduser("~/push_logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"push_{timestamp}.csv")

        # Write CSV header
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "duration",
                "vx", "vy", "vz", "wx", "wy", "wz",
                "avg_fx", "avg_fy", "avg_fz",
                "avg_tx", "avg_ty", "avg_tz",
                "n_samples"
            ])

    def joint_state_callback(self, msg: JointState):
        joint_map = {name: pos for name, pos in zip(msg.name, msg.position)}
        if all(name in joint_map for name in self.joint_names):
            self.joint_positions = np.array([joint_map[name] for name in self.joint_names])
            self.motion.update_joint_positions(self.joint_positions)

    def ee_pose_callback(self, msg: PoseStamped):
        self.ee_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                 msg.pose.orientation.z, msg.pose.orientation.w])
        self.ee_euler = canonicalize_euler(R.from_quat(self.ee_quat).as_euler('xyz', degrees=True))
        self.gripper.update(self.ee_position, VERTICAL_OFFSET)
        self.pusher.update(self.ee_position, self.ee_euler)

    def wrench_callback(self, msg: WrenchStamped):
        raw_force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        # print(f"RAW  {raw_force}")
        raw_torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        if self.filtered_force is None or self.filtered_torque is None:
            self.filtered_force = raw_force
            self.filtered_torque = raw_torque
        # Apply exponential smoothing
        self.filtered_force = self.alpha * raw_force + (1 - self.alpha) * self.filtered_force
        self.filtered_torque = self.alpha * raw_torque + (1 - self.alpha) * self.filtered_torque
        # print(f"FILT {self.filtered_force}")

        # Offset initialization
        if self.wrench_offset_initialized:
            self.wrench.update(self.force_offset, self.torque_offset, self.filtered_force, self.filtered_torque)
            self.adjusted_force = self.filtered_force - self.force_offset
            self.adjusted_torque = self.filtered_torque - self.torque_offset

    def initialize_wrench_offset(self, wait_period=3.0, measure_period=3.0):
        """
        Keeps the robot still and initializes the force offset. Waits for wait_period to make the first measure, then measures for measure_period.
        """
        self.get_logger().info(f"Initializing force sensor offset for {wait_period + measure_period} seconds. Keep the robot still...")
        
        self.offset_force_accumulator = [] # Reset accumulators if not empty
        self.offset_torque_accumulator = []

        start_time = self.get_clock().now().nanoseconds / 1e9
        last_progress = start_time
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            now = self.get_clock().now().nanoseconds / 1e9
            elapsed = now - start_time
            # Progress printouts
            if now - last_progress >= 1.0:
                print(
                    f"Waiting: {elapsed:.1f}s elapsed"
                )
                last_progress = now

            if elapsed < wait_period:
                continue # Do nothing
            # Accumulate filtered wrench readings
            self.offset_force_accumulator.append(self.filtered_force)
            self.offset_torque_accumulator.append(self.filtered_torque)
            mean_force_offset = np.mean(self.offset_force_accumulator, axis=0)
            mean_torque_offset = np.mean(self.offset_torque_accumulator, axis=0)
            # print(f"Measuring force offset as: {vector3_text(mean_force_offset)}")
            if elapsed > wait_period + measure_period:
                self.force_offset = mean_force_offset
                self.torque_offset = mean_torque_offset
                self.wrench_offset_initialized = True
                self.get_logger().info(f"Force offset initialized: {self.force_offset}")
                break

    def record_push_segment(self, velocity_cmd: np.ndarray, duration: float,
                            threshold: float = 0.01, settle_time: float = 0.5):
        """
        Send a velocity command and measure average force/torque response.

        Args:
            velocity_cmd (np.ndarray): 6D velocity command.
            duration (float): Duration of velocity motion (sec).
            threshold (float): Minimum change in |force| (N) to detect push onset.
            settle_time (float): Extra time to wait after command end for relaxation.

        Returns:
            dict: {
                'avg_force': np.ndarray,
                'avg_torque': np.ndarray,
                'force_samples': list,
                'torque_samples': list
            }
        """

        force_samples, torque_samples = [], []
        collecting = True
        onset_detected = False

        def collector():
            nonlocal onset_detected
            while collecting and rclpy.ok():
                if self.adjusted_force is None or self.adjusted_torque is None:
                    time.sleep(0.01)
                    continue

                f = self.adjusted_force
                print(f"FORCE {vector3_text(f)}")
                t = self.adjusted_torque

                # Detect onset
                if not onset_detected and np.linalg.norm(f) > threshold:
                    onset_detected = True
                    self.get_logger().info("Force Detected")

                if onset_detected:
                    force_samples.append(f.copy())
                    torque_samples.append(t.copy())

                time.sleep(0.01)

        # Start logging thread
        t = threading.Thread(target=collector)
        t.start()

        # Run the blocking velocity command
        self.motion.send_cartesian_velocity(velocity_cmd, duration=duration)

        # Let things settle after motion
        time.sleep(settle_time)

        # Stop collector
        collecting = False
        t.join()

        # Compute averages
        avg_force = np.mean(force_samples, axis=0) if force_samples else np.zeros(3)
        avg_torque = np.mean(torque_samples, axis=0) if torque_samples else np.zeros(3)

        # --- Save to CSV log ---
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                duration,
                *velocity_cmd.tolist(),
                *avg_force.tolist(),
                *avg_torque.tolist(),
                len(force_samples)
            ])

        self.get_logger().info(
            f"Push done. Avg Force: {vector3_text(avg_force)}, "
            f"Avg Torque: {vector3_text(avg_torque)}, "
            f"samples: {len(force_samples)}"
        )

        return {
            'avg_force': avg_force,
            'avg_torque': avg_torque,
            'force_samples': force_samples,
            'torque_samples': torque_samples
        }



def main(args=None):
    rclpy.init(args=args)
    node = MixedCartesianController()


    # Go to home
    home_pos = [0.1, -0.5, 0.181]
    home_rot = [0, 180, 0]
    repos_seconds = 2
    node.motion.send_cartesian_position(home_pos, home_rot, seconds=4)


    for i in range(3):
        node.initialize_wrench_offset()
        push = node.record_push_segment(np.array([0.0, -0.0005, 0.0, 0.0, 0.0, 0.0]), duration=20.0) #  10mm
        back = node.record_push_segment(np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]), duration=1.0) #  10mm
        node.motion.send_cartesian_position(home_pos, home_rot, seconds=repos_seconds)

        node.initialize_wrench_offset()
        push = node.record_push_segment(np.array([0.0, -0.0010, 0.0, 0.0, 0.0, 0.0]), duration=20.0) #  20mm
        back = node.record_push_segment(np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]), duration=2.0) #  20mm
        node.motion.send_cartesian_position(home_pos, home_rot, seconds=repos_seconds)

        node.initialize_wrench_offset()
        push = node.record_push_segment(np.array([0.0, -0.0020, 0.0, 0.0, 0.0, 0.0]), duration=20.0) #  40mm
        back = node.record_push_segment(np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0]), duration=2.0) #  40mm
        node.motion.send_cartesian_position(home_pos, home_rot, seconds=repos_seconds)

        node.initialize_wrench_offset()
        push = node.record_push_segment(np.array([0.0, -0.0050, 0.0, 0.0, 0.0, 0.0]), duration=20.0) # 100mm
        back = node.record_push_segment(np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0]), duration=5.0) # 100mm
        node.motion.send_cartesian_position(home_pos, home_rot, seconds=repos_seconds)

        node.initialize_wrench_offset()
        push = node.record_push_segment(np.array([0.0, -0.0100, 0.0, 0.0, 0.0, 0.0]), duration=10.0) # 100mm
        back = node.record_push_segment(np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0]), duration=5.0) # 100mm
        node.motion.send_cartesian_position(home_pos, home_rot, seconds=repos_seconds)
        # print(f"PUSH {i}: {vector3_text(push['avg_force'])}")

    # Long pause to allow testing of sensor
    node.motion.send_cartesian_position(home_pos, home_rot, seconds=400)


    node.get_logger().info("Mixed Cartesian motion complete.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
