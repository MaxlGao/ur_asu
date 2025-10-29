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
        
        # Targeting
        # Options: jenga_###, allen_key, wrench
        self.target_name = "wrench"
        print(f"Looking for object {self.target_name}")
        self.object_subscription = self.create_subscription(
            PoseStamped, f"/object_poses/{self.target_name}", self.object_pose_callback, 10
            )
        self.latest_target_pose = None
        self.previous_target_pose = None
        self.last_pose_time = time.time()
        self.pose_timeout = 1.0

        # Motion Planning
        start_duration = 3
        start_position = [0.00, -0.550, GRIPPER_TABLE[0] + self.vert_offset + 0.05]
        start_orientation = [0, 180, 0]
        first_move = move(start_position, start_orientation, start_duration)
        self.acts = {}
        self.acts = append_new_act(self.acts, first_move)

        self.active_goal_handle = None
        self.executing = False
        self.i = 0
        self.execute_next_act()

    def execute_next_act(self):
        if self.i >= len(self.acts):
            self.get_logger().info("Done with current list. Waiting for more...")
            self.executing = False
            self.check_for_new_pose()
            return
            # if False: # some exit criteria
            #     self.get_logger().info("Done with motion planning")
            #     raise SystemExit
        else:
            act_name = list(self.acts)[self.i]
            self.i += 1
            self.execute_act(act_name)

    def execute_act(self, act_name):
        self.executing = True  # Flag it
        act = self.acts[act_name]
        act_type = act["type"]
        self.get_logger().info(f"▶ Executing {act_name}: {act_type}")
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
        self.execute_next_act()

    def object_pose_callback(self, msg: PoseStamped):
        xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        quat = (msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w
        )
        rpy = R.from_quat(quat).as_euler('xyz', degrees=True)
        self.latest_target_pose = (xyz, rpy)
        self.last_pose_time = time.time()

    def check_for_new_pose(self):
        if (time.time() - self.last_pose_time) > self.pose_timeout:
            if not self.executing:
                self.get_logger().warn("Lost target — initiating search pattern - STAND CLEAR.")
                self.run_search_pattern()
            return

        if self.latest_target_pose is None:
            # No pose available
            return
        
        if self.previous_target_pose is None or self._pose_changed_enough():
            if len(self.motion.ee_euler) == 0:
                print("No EE data; cannot estimate object pose!")
                return
            self.get_logger().info(f"New target detected!")
            self.update_act()
            self.previous_target_pose = self.latest_target_pose

    def _pose_changed_enough(self, lin_threshold=0.01, ang_threshold=10):
        if self.previous_target_pose is None:
            return True
        old_pos, old_rot = self.previous_target_pose
        new_pos, new_rot = self.latest_target_pose
        lin_dist = np.linalg.norm(np.array(old_pos) - np.array(new_pos))
        rot_dist = np.linalg.norm(np.array(old_rot) - np.array(new_rot))
        return lin_dist > lin_threshold or rot_dist > ang_threshold

    def update_act(self, segment_duration=3):
        """
        This is a pointspan generation function that makes a list
        of acts, bringing the grippers safely to their target.
        We assume that there exist actual recommended points and corresponding normal vectors. 
        This will lift the gripper up, move it over and away from the target,
        bring it down slowly, and then move it slowly to just touch the object.
        """ 
        pusher_1, pusher_2, given_yaw = self.pusher.extract_pusher_locations()
        goal_pos, goal_ori = self.gripper.pushers_to_EE_pose_2D(pusher_1, pusher_2, given_yaw)
        pusher_1, span, yaw = self.gripper.EE_pose_to_pointspan([goal_pos, goal_ori])

        moves = self.gripper.pointspan_to_acts_safe(
            [self.motion.ee_position, self.motion.ee_euler], 
            pusher_1, span, yaw, segment_duration)
        
        push_yaw = goal_ori[2] - 90
        push_yaw_r = np.deg2rad(push_yaw)
        frce = 5
        push_vector = [frce*np.cos(push_yaw_r), frce*np.sin(push_yaw_r), np.float64(0), np.float64(0), np.float64(0), np.float64(0)]
        push = force(push_vector, 5.0)
        moves.append(push)
        # flush queue and start
        self.acts = {}
        self.i = 0 
        for mov in moves:
            self.acts = append_new_act(self.acts, mov)
        self.execute_next_act()   

    def run_search_pattern(self):
        current_pos = np.array(self.motion.ee_position)
        current_ori = np.array(self.motion.ee_euler)
        target_pos = current_pos + np.array([0, 0, 0.15])
        lift = move(
            (target_pos).tolist(),
            current_ori.tolist(),
            4
        )
        self._execute_single_act(lift)
        self.get_logger().info("Search pattern complete — waiting for detections.")
        
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
