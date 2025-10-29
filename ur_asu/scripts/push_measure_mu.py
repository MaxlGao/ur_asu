import time
import numpy as np
import rclpy
import csv
import threading
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from ur_asu.custom_libraries.motion_utils import MotionExecutor
from ur_asu.custom_libraries.pusher_utils import PusherHandler
from ur_asu.custom_libraries.gripper_utils import GripperHandler, GRIPPER_TABLE
from ur_asu.custom_libraries.actionlibrariesmax import move, velocity, force, gripper_width

class OverController(Node):
    def __init__(self):
        super().__init__('motion_planner')

        # Controls
        self.sticky = False                # Two friction modes: Sticky or Smooth
        self.linear = True                 # Linear or rotational testing
        self.target_name = "jenga_6"        # Options: jenga_6, wrench


        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        #region setup
        # Controller Names
        self.position_controller = 'scaled_joint_trajectory_controller'
        self.velocity_controller = 'forward_velocity_controller'
        self.force_controller = 'force_mode_controller'
        self.passthrough_controller = 'passthrough_trajectory_controller'

        # Subclasses
        self.pusher = PusherHandler(self)
        self.vert_offset = 0.003
        self.gripper = GripperHandler(self, vertical_offset = self.vert_offset)
        self.motion = MotionExecutor(
            self, self.joint_names,
            self.position_controller,
            self.velocity_controller,
            self.force_controller,
            self.passthrough_controller,
            self.gripper,
            pusher=self.pusher,
            auto_gripper = False
            )
        
        self.push_duration = 6.0          # seconds per trial
        self.pause_between = 3.0          # seconds between pushes
        self.log_data = []                # store (time, Fy, x, y, ee_x, ee_y)
        self.start_time = None

        print(f"Looking for object {self.target_name}")
        self.object_subscription = self.create_subscription(
            PoseStamped, f"/object_poses/{self.target_name}", self.object_pose_callback, 10
            )
        self.latest_target_pose = None

        if self.target_name == "wrench":
            self.gripper_width = 1100
        elif self.target_name == "jenga_6":
            self.gripper_width = 400

        # Two regions on the table with different friction coeffs
        if self.sticky:
            adj1 = "sticky"
            self.home_position = [-0.361, -0.310, GRIPPER_TABLE[self.gripper_width] + self.vert_offset]
            self.force_levels = [2.0, 4.0, 6.0, 8.0, 10.0]  # [N]
            self.torque_levels = [0.2, 0.4, 0.6, 0.8, 1.0]  # [N]
        else:
            adj1 = "smooth"
            self.home_position = [-0.120, -0.400, GRIPPER_TABLE[self.gripper_width] + self.vert_offset]
            self.force_levels = [0.2, 0.5, 1.0, 2.0, 4.0]  # [N]
            self.torque_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # [N]

        if self.target_name == "wrench":
            # Wrench is about 7x heavier than jenga
            self.force_levels = [f*3 for f in self.force_levels]

        self.base_orientation = [0, 180, 0]

        if self.linear:
            adj2 = "linear"
        else:
            adj2 = "rotational"
            # bump y position by a bit, since we're not pushing that way
            self.home_position[1] += -0.100

        #endregion

        self.logfile = f"friction_trials_{adj1}_{self.target_name}_{adj2}.csv"
        print(f"Starting {self.logfile}")

        gripper_open_act = gripper_width('1100') # Fully open gripper before moving
        self._execute_single_act(gripper_open_act)
        if self.sticky:
            first_move_act = move(self.home_position, self.base_orientation, 5)
        else:
            first_move_act = move(self.home_position, self.base_orientation, 5)

        self._execute_single_act(first_move_act)
        gripper_adjust_act = gripper_width(str(self.gripper_width))
        self._execute_single_act(gripper_adjust_act)

        for t in range(5):
            print(f"Starting in {5-t}...")
            time.sleep(1)

        self.run_friction_experiment()

    def object_pose_callback(self, msg: PoseStamped):
        xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        quat = (msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w)
        rpy = R.from_quat(quat).as_euler('xyz', degrees=True)
        self.latest_target_pose = (xyz, rpy)

    def _log_pose_callback(self):
        if self.latest_target_pose is not None and self.current_force is not None:
            t = time.time() - self.start_time
            x, y, _ = self.latest_target_pose[0]
            ee_x, ee_y, _ = self.motion.ee_position
            self.log_data.append([t, self.current_force, x, y, ee_x, ee_y])

    def run_friction_experiment(self):
        self.get_logger().info("Starting kinetic friction experiment")
        with open(self.logfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time","Fy","x","y","ee_x","ee_y"])
            # if self.linear:
            #     levels = self.force_levels
            # else:
            #     levels = self.torque_levels

            for Fy_cmd in self.force_levels:
                self.get_logger().info(f"→ Push with {Fy_cmd:.1f} N")
                self.start_time = time.time()
                self.current_force = Fy_cmd
                self.log_data = []

                push_vec = [0.0, -Fy_cmd, 0.0, 0.0, 0.0, 0.0]
                print(push_vec)
                # torq_vec = [0.0, 0.0, 0.0, 0.0, 0.0, Tz_cmd]
                push_act = force(push_vec, self.push_duration, selection_vector=[False, True, False, False, False, False])
                self._execute_single_act(push_act)

                while time.time() - self.start_time < self.push_duration:
                    self._log_pose_callback()
                    time.sleep(0.02)

                # save to disk
                for row in self.log_data:
                    writer.writerow(row)
                f.flush()

                # Reset position, if needed
                # go_home = move(self.home_position, self.base_orientation, 3.0)
                # self._execute_single_act(go_home)

                # short pause
                for t in range(int(self.pause_between)):
                    print(f"Starting in {self.pause_between-t}...")
                    time.sleep(1)

        self.get_logger().info("Friction experiment complete.")
        self.start_time = None
        self.current_force = None

    def _execute_single_act(self, act):
        """Execute a single act immediately (non-queued)."""
        act_type = act["type"]

        if act_type == "position":
            joint_angles = act["joint_angles"]
            seconds = act["time_from_start"]
            self.motion.send_joint_angles(joint_angles, seconds)
        elif act_type == "velocity":
            velocity = act["velocity"]
            seconds = act["duration"]
            self.motion.send_cartesian_velocity(velocity, seconds)
        elif act_type == "force":
            force = act["force"]
            seconds = act["duration"]
            # selection vector is x y z rx ry rz boolean to permit movement in the given axis
            selection_vector = act["selection_vector"]
            self.motion.send_cartesian_force_async(force, seconds, selection_vector=selection_vector)
        elif act_type == "gripper":
            cmd = act["cmd"]
            self.gripper.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = OverController()

    try:
        rclpy.spin(node)
    except (RuntimeError, SystemExit):
        node.get_logger().info("Shutting down")
    node.get_logger().info("Script complete.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
