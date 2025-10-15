import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from ur_asu.custom_libraries.motion_utils import MotionExecutor
from ur_asu.custom_libraries.pusher_utils import PusherHandler
from ur_asu.custom_libraries.gripper_utils import GripperHandler, GRIPPER_TABLE
from ur_asu.custom_libraries.actionlibrariesmax import move, velocity, force, append_new_act

def make_circle_velocity_traj(radius=0.08, total_time=10.0, steps=200):
    omega = 2 * np.pi / total_time
    v_mag = radius * omega
    dt = total_time / steps
    traj = []
    for i in range(steps):
        theta = i * 2 * np.pi / steps
        vx = -v_mag * np.sin(theta)
        vy = v_mag * np.cos(theta)
        traj.append(np.array([vx, vy, 0.0, 0.0, 0.0, 0.0]))
    return traj, dt

def make_circle_force_traj(radius=0.05, total_time=10.0, steps=100, force_mag=10.0):
    dt = total_time / steps
    traj = []
    for i in range(steps):
        theta = i * 2 * np.pi / steps
        fx = -force_mag * np.sin(theta)
        fy =  force_mag * np.cos(theta)
        traj.append([fx, fy, 0.0, 0.0, 0.0, 0.0])
    return traj, dt

class OverController(Node):
    def __init__(self):
        super().__init__('motion_planner')

        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

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
            pusher=self.pusher
            )

        # Motion Planning
        move_1 = move([0.1, -0.5, 0.247], [0, 180, 0], 4.0)
        # move_2 = velocity(np.array([0.0, -0.02, 0.0, 0.0, 0.0, 0.0]), 5.0)
        # move_3 = velocity(np.array([0.0,  0.02, 0.0, 0.0, 0.0, 0.0]), 5.0)
        # move_4 = force([0.0, 0.0, -10.0, 0.0, 0.0, 0.0], 3.0)
        self.acts = {}
        self.acts = append_new_act(self.acts, move_1)
        # self.acts = append_new_act(self.acts, move_2)
        # self.acts = append_new_act(self.acts, move_3)
        # self.acts = append_new_act(self.acts, move_4)

        # Circle motion using velocity commands
        # traj, dt = make_circle_velocity_traj()
        # self.motion.send_cartesian_velocity_trajectory(traj, dt)

        # Circle motion using force commands (optional or separate run)
        traj, dt = make_circle_force_traj()
        self.motion.send_cartesian_force_trajectory(traj, dt)


        self.active_goal_handle = None
        self.executing = False
        self.i = 0
        self.execute_next_act()

    def execute_next_act(self):
        if self.i >= len(self.acts):
            self.get_logger().info("Done with current list. Waiting for more...")
            self.executing = False
            return
            # if False: # some exit criteria
            #     self.get_logger().info("Done with motion planning")
            #     raise SystemExit
        else:
            act_name = list(self.acts)[self.i]
            self.i += 1
            self.execute_act(act_name)

    def execute_act(self, act_name):
        """Execute one act, then the next"""
        act = self.acts[act_name]
        act_type = act["type"]
        self.get_logger().info(f"▶ Executing {act_name}: {act_type}")
        self._execute_single_act(act)
        self.execute_next_act() 

    def _execute_single_act(self, act):
        """Execute a single act immediately (non-queued)."""
        act_type = act["type"]
        self.executing = True
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
            self.motion.send_cartesian_force(force, seconds)
        elif act_type == "gripper":
            cmd = act["cmd"]
            self.gripper.publish(cmd)
        else:
            raise ValueError("Invalid Act Type")
        
        self.executing = False




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
