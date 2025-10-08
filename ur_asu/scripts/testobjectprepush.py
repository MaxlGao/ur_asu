import time
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from builtin_interfaces.msg import Duration
from action_msgs.msg import GoalStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped
from scipy.spatial.transform import Rotation as R
from functools import partial
import argparse

from ur_asu.custom_libraries.actionlibrariesmax import move, hover_over
from ur_asu.custom_libraries.gripperlibraries import *

# Script allows for gripper positions to be sent as inputs. 
GRIPPER_FINGER_OFFSET = 0.0058 # m

def canonicalize_euler(orientation):
    """Forces euler angles near the form (-180, 0, yaw') to take the equivalent form (0, 180, yaw)"""
    roll, pitch, yaw = orientation
    if abs(pitch) < 1 and abs(abs(roll) - 180) < 1:
        return (0.0, 180.0, (yaw % 360)-180)
    else:
        return orientation

def pose_text(pose):
    position, euler = pose
    return f"""XYZ: {1000*position[0]:.1f}, {1000*position[1]:.1f}, {1000*position[2]:.1f} mm
RPY: {euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f} deg"""

def pushers_text(pusher_1_pos, pusher_2_pos):
    return f"""
    Pusher 1: ({1000*pusher_1_pos[0]:.1f}, {1000*pusher_1_pos[1]:.1f}, {1000*pusher_1_pos[2]:.1f}) mm
    Pusher 2: ({1000*pusher_2_pos[0]:.1f}, {1000*pusher_2_pos[1]:.1f}, {1000*pusher_2_pos[2]:.1f}) mm"""

def pointspan_to_trajectories_safer(EE_pose_now, pusher_1_target, span_target, yaw_target, duration):
    """
    Same as above, now with retract, raise, move, lower, and advance actions. 
    Fortunately we only need the EE pose at the first instant. 
    """
    print(f"""We're currently at:
{pose_text(EE_pose_now)}""")
    moves = []
    # Part 1: Retreat. Find the yaw, and set a waypoint 1 cm backward
    retreat_distance = 0.03
    initial_pos, initial_ori = EE_pose_now
    # At 0 yaw, the retreat direction is towards +90 degrees.
    retreat_angle = initial_ori[2] + 90
    retreat_angle_r = np.deg2rad(retreat_angle)
    retreat_dx = retreat_distance * np.cos(retreat_angle_r)
    retreat_dy = retreat_distance * np.sin(retreat_angle_r)
    retreat_x = initial_pos[0] + retreat_dx
    retreat_y = initial_pos[1] + retreat_dy
    retreat_position = [retreat_x, retreat_y, initial_pos[2]]
    retreat_pose = [retreat_position, initial_ori]
    moves.append(move(retreat_position, initial_ori, duration))
    print(f"""RETREAT:
{pose_text(retreat_pose)}""")

    # Part 2: Raise. Move to a height of 300mm
    raise_position = retreat_position
    raise_position[2] = 0.3 + VERTICAL_OFFSET
    print(raise_position)
    initial_width = height_to_gripper_width(initial_pos[2] - VERTICAL_OFFSET) * 0.0001
    if initial_width > 0.04:
        intermed_height = gripper_width_to_height(400) + VERTICAL_OFFSET
        intermed_position = raise_position.copy()
        intermed_position[2] = intermed_height
        intermed_pose = [intermed_position, initial_ori]
        print(f"""INTERMED 1:
{pose_text(intermed_pose)}""")
        moves.extend([move(intermed_position, initial_ori, duration),
                      move(raise_position, initial_ori, 2*duration)])
    else:
        moves.append(move(raise_position, initial_ori, 2*duration))

    raise_pose = [raise_position, initial_ori]
    print(f"""RAISE:
{pose_text(raise_pose)}""")


    # Part 5: Advance. Move above a final position. 
    # We calculate this first, then back calculate the other steps.
    end_pos, end_ori = pointspan_to_EE_pose_2D(pusher_1_target, span_target, yaw_target)
    # For the end position we must consider the gripper width. 
    # So, we repeat some of Part 1's retreat. 
    advance_angle = end_ori[2] + 90
    advance_angle_r = np.deg2rad(advance_angle)
    end_dx = GRIPPER_FINGER_OFFSET * np.cos(advance_angle_r)
    end_dy = GRIPPER_FINGER_OFFSET * np.sin(advance_angle_r)
    end_x = end_pos[0] + end_dx
    end_y = end_pos[1] + end_dy
    end_position = [end_x, end_y, end_pos[2]]
    end_moves = [move(end_position, end_ori, duration)] # To be reversed later
    # We assume that the camera side is facing the object, so moving forward means approaching.
    advance_distance = retreat_distance
    advance_dx = advance_distance * np.cos(advance_angle_r)
    advance_dy = advance_distance * np.sin(advance_angle_r)
    advance_x = end_pos[0] + advance_dx
    advance_y = end_pos[1] + advance_dy
    advance_position = [advance_x, advance_y, end_pos[2]]
    end_moves.append(move(advance_position, end_ori, duration))
    
    # Part 4: Lower. Again, reverse Part 2.
    lower_position = advance_position
    lower_position[2] = 0.3 + VERTICAL_OFFSET
    print(lower_position)

    end_width = height_to_gripper_width(end_pos[2] - VERTICAL_OFFSET) * 0.0001
    if end_width > 0.04:
        intermed_height = gripper_width_to_height(400) + VERTICAL_OFFSET
        intermed_position = lower_position.copy()
        intermed_position[2] = intermed_height
        end_moves.extend([move(intermed_position, end_ori, duration),
                      move(lower_position, end_ori, 2*duration)])
    else:
        end_moves.append(move(lower_position, end_ori, 2*duration))
    lower_pose = [lower_position, end_ori]
    print(f"""SLIDE:
{pose_text(lower_pose)}""")
    print("INTERMED 2: it probably exists")
    advance_pose = [advance_position, end_ori]
    print(f"""LOWER:
{pose_text(advance_pose)}""")
    end_pose = [end_pos, end_ori]
    print(f"""END:
{pose_text(end_pose)}""")

    # Part 3: Translate. Simply join the two lists together
    moves.extend(reversed(end_moves))
    return moves


def append_new_traj(traj_dict, trajectory):
    # Extract the numeric suffixes from keys and find the max
    existing_keys = [key for key in traj_dict.keys() if key.startswith("traj")]
    if existing_keys:
        max_index = max(int(key[4:]) for key in existing_keys if key[4:].isdigit())
    else:
        max_index = 0

    new_key = f"traj{max_index + 1}"
    traj_dict[new_key] = trajectory
    return traj_dict


class JTCClient(Node):
    def __init__(self, **kwargs):
        super().__init__("trajectory_executor", **kwargs)
        # Parameter Management
        self.declare_parameter("controller_name", "scaled_joint_trajectory_controller")
        self.declare_parameter("joints", [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"])
        
        # OBJECT TO FOLLOW
        # Options: jenga_###, allen_key, wrench
        self.declare_parameter("target", "wrench")

        self.ee_position = []
        self.ee_quat = []
        self.ee_euler = []

        controller_name = self.get_parameter("controller_name").value + "/follow_joint_trajectory"
        self.joints = self.get_parameter("joints").value
        self.target_object_name = self.get_parameter("target").value
        print(f"Looking for object {self.target_object_name}")

        self._action_client = ActionClient(self, FollowJointTrajectory, controller_name)
        self._gripper_pub = self.create_publisher(String, "/gripper_command", 10)
        self.object_subscription = self.create_subscription(
            PoseStamped,
            f"/object_poses/{self.target_object_name}",  # Listening to a specific object
            self.object_pose_callback,
            10)

        self.ee_subscription = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            10)

        self.recommended_topics = [
            ('pusher_1_position', '/recommended_pusher_1/position', PointStamped),
            ('pusher_2_position', '/recommended_pusher_2/position', PointStamped),
            ('pusher_1_normal',   '/recommended_pusher_1/normal',   Vector3Stamped),
            ('pusher_2_normal',   '/recommended_pusher_2/normal',   Vector3Stamped)
        ]
        self.recommended_subscriptions = {}

        for name, topic, msg_type in self.recommended_topics:
            callback_with_name = partial(self.push_recommend_callback, name=name)
            sub = self.create_subscription(msg_type, topic, callback_with_name, 10)
            self.recommended_subscriptions[name] = sub
        self.recommend_data = {
            "pusher_1": {"position": None, "normal": None},
            "pusher_2": {"position": None, "normal": None},
        }

        self.pusher_pub_l = self.create_publisher(PointStamped, "/gripper_pusher_left", 10)
        self.pusher_pub_r = self.create_publisher(PointStamped, "/gripper_pusher_right", 10)


        self.get_logger().info(f"Waiting for action server on {controller_name}")
        self._action_client.wait_for_server()

        self.latest_target_pose = None
        self.last_sent_pose = None
        # self.pose_update_time = time.time()

        # Timer to check for new poses periodically
        self.timer = self.create_timer(1.0, self.check_for_new_pose)  # every 1s
        
        start_duration = 3
        self.segment_duration = 3 # 1.5 is a mite aggressive. 

        start_position = [0.00, -0.550, GRIPPER_TABLE[0] + VERTICAL_OFFSET]
        start_orientation = [0, 180, 0]
        first_move = move(start_position, start_orientation, start_duration)
        self.trajectories = {}
        self.trajectories = append_new_traj(self.trajectories, first_move)

        self.active_goal_handle = None
        self.executing = False
        self.last_pose_sent_time = 0
        self.goals = self.parse_trajectories()
        self.pointspan_index = 0
        self.i = 0
        self.execute_next_trajectory()


    def ee_pose_callback(self, msg):
        # Currently publishes gripper based on subscribed z coordinate
        # So, if you want to set the gripper to some value, make the EE go down more.
        self.ee_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                     msg.pose.orientation.z, msg.pose.orientation.w])
        ee_euler = R.from_quat(self.ee_quat).as_euler('xyz', degrees=True)
        self.ee_euler = canonicalize_euler(ee_euler)

        # Remove offset to get height above virtual table
        actual_height = self.ee_position[2] - VERTICAL_OFFSET

        gripper_width = height_to_gripper_width(actual_height)
        self.publish_gripper(str(gripper_width))

        pusher_1_pos, pusher_2_pos = EE_pose_to_pushers_2D((self.ee_position, self.ee_euler))
        # print(f"Pusher 1: ({1000*pusher_1_pos[0]:.1f}, {1000*pusher_1_pos[1]:.1f}, {1000*pusher_1_pos[2]:.1f}) mm")
        # print(f"Pusher 2: ({1000*pusher_2_pos[0]:.1f}, {1000*pusher_2_pos[1]:.1f}, {1000*pusher_2_pos[2]:.1f}) mm")

        now = self.get_clock().now().to_msg()
        pos_msg_l = PointStamped()
        pos_msg_l.header.stamp = now
        pos_msg_l.header.frame_id = "base"
        pos_msg_l.point.x, pos_msg_l.point.y, pos_msg_l.point.z = pusher_1_pos
        self.pusher_pub_l.publish(pos_msg_l)
        
        pos_msg_r = PointStamped()
        pos_msg_r.header.stamp = now
        pos_msg_r.header.frame_id = "base"
        pos_msg_r.point.x, pos_msg_r.point.y, pos_msg_r.point.z = pusher_2_pos
        self.pusher_pub_r.publish(pos_msg_r)

    def push_recommend_callback(self, msg, name):
        # Partial function, since this gets called for EVERY recommend subscriber (there are 4, and they publish one after another)
        # self.get_logger().info(f"Received {name}: {msg}")
        _, number, attr = name.split("_")
        self.recommend_data[f"pusher_{number}"][attr] = msg


    def parse_trajectories(self):
        goals = {}
        for traj_name, points in self.trajectories.items():
            traj = JointTrajectory()
            traj.joint_names = self.joints
            for pt in points:
                point = JointTrajectoryPoint()
                point.positions = pt["positions"]
                point.velocities = pt["velocities"]
                point.time_from_start = pt["time_from_start"]
                traj.points.append(point)
            goals[traj_name] = traj
        return goals

    def publish_gripper(self, cmd):
        msg = String()
        msg.data = cmd
        # self.get_logger().info(f"Sending gripper command: {cmd}")
        self._gripper_pub.publish(msg)

    def execute_next_trajectory(self):
        if self.i >= len(self.goals):
            self.get_logger().info("Done with current list")
        else:
            traj_name = list(self.goals)[self.i]
            self.i += 1

            self.execute_trajectory(traj_name)

    def execute_trajectory(self, traj_name):
        self.get_logger().info(f"▶ Executing trajectory {traj_name}")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self.goals[traj_name]
        goal.goal_time_tolerance = Duration(sec=0, nanosec=500_000_000)
        goal.goal_tolerance = [JointTolerance(position=0.01, velocity=0.01, name=name) for name in self.joints]

        self.executing = True  # Flag it
        self._send_goal_future = self._action_client.send_goal_async(goal)
        self._send_goal_future.add_done_callback(lambda f: self.goal_response_callback(f, traj_name))


    def object_pose_callback(self, msg: PoseStamped):
        # Convert quaternion to rpy
        quat = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        rpy = R.from_quat(quat).as_euler('xyz', degrees=True)
        xyz = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]
        self.latest_target_pose = (xyz, rpy)

    def check_for_new_pose(self):
        if self.latest_target_pose is None:
            return

        # Only replan if we're not running something
        if self.executing:
            return

        if self.last_sent_pose is None or self._pose_changed_enough():
            if len(self.ee_euler) == 0:
                print("No EE data!")
                return
            self.get_logger().info(f"New target detected!")
            self.update_trajectory()
            self.last_sent_pose = self.latest_target_pose

    def _pose_changed_enough(self, lin_threshold=0.01, ang_threshold=10):
        if self.last_sent_pose is None:
            return True
        old_pos, old_rot = self.last_sent_pose
        new_pos, new_rot = self.latest_target_pose
        lin_dist = np.linalg.norm(np.array(old_pos) - np.array(new_pos))
        rot_dist = np.linalg.norm(np.array(old_rot) - np.array(new_rot))
        return lin_dist > lin_threshold or rot_dist > ang_threshold

    def update_trajectory(self):
        """
        This is a pointspan generation function that makes a list
        of lists of trajectories, bringing the grippers safely to their target.
        We assume that there exist actual recommended points and corresponding normal vectors. 
        This will lift the gripper up, move it over and away from the target,
        bring it down slowly, and then move it slowly to just touch the object.
        """ 
        # If any of recommended pushers are none, skip this process
        if any([self.recommend_data["pusher_1"]["position"] is None,
                self.recommend_data["pusher_2"]["position"] is None,
                self.recommend_data["pusher_1"]["normal"] is None,
                self.recommend_data["pusher_2"]["normal"] is None]):
            print("Improper data detected. Skipping...")
            return

        # Find the EE Pose from pushers
        # normals are normal out. We want that to be anti-camera, which is +90deg
        # normals are also unit vector. 
        x, y, z = [
            self.recommend_data["pusher_1"]["normal"].vector.x,
            self.recommend_data["pusher_1"]["normal"].vector.y,
            self.recommend_data["pusher_1"]["normal"].vector.z
        ]
        normal_yaw = np.arctan2(y, x)  # radians
        normal_yaw_d = np.degrees(normal_yaw)
        given_yaw = normal_yaw_d - 90
        pusher_1 = [
            self.recommend_data["pusher_1"]["position"].point.x,
            self.recommend_data["pusher_1"]["position"].point.y,
            self.recommend_data["pusher_1"]["position"].point.z
        ]
        pusher_2 = [
            self.recommend_data["pusher_2"]["position"].point.x,
            self.recommend_data["pusher_2"]["position"].point.y,
            self.recommend_data["pusher_2"]["position"].point.z
        ]

        goal_pos, goal_ori = pushers_to_EE_pose_2D(pusher_1, pusher_2, given_yaw)
        # Then get pointspan
        pusher_1, span, yaw = EE_pose_to_pointspan([goal_pos, goal_ori])

        moves = pointspan_to_trajectories_safer([self.ee_position, self.ee_euler], 
                                        pusher_1, span, yaw, self.segment_duration)
        self.trajectories = {}
        for mov in moves:
            self.trajectories = append_new_traj(self.trajectories, mov)
        self.i = 0 # restart
        self.goals = self.parse_trajectories()
        self.execute_next_trajectory()

    def goal_response_callback(self, future, traj_name):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected :(")
            raise RuntimeError("Goal rejected :(")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(lambda f: self.get_result_callback(f, traj_name))

    def get_result_callback(self, future, traj_name):
        result = future.result().result
        status = future.result().status
        # self.get_logger().info(f"✔ Trajectory {traj_name} completed with status: {self.status_to_str(status)}")

        self.executing = False  # <-- Reset flag

        if status == GoalStatus.STATUS_SUCCEEDED:
            time.sleep(0)
            self.execute_next_trajectory()
        else:
            raise RuntimeError("Trajectory failed: " + str(result.error_string))

    @staticmethod
    def status_to_str(status):
        return {
            GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
            GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
            GoalStatus.STATUS_EXECUTING: "EXECUTING",
            GoalStatus.STATUS_CANCELING: "CANCELING",
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
        }.get(status, "?")

def main(args=None):
    rclpy.init(args=args)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the JTCClient node with a specified target object.")
    parser.add_argument('--target', type=str, default='allen_key', help='Name of the target object to follow')
    parsed_args, unknown = parser.parse_known_args()
    print(f"Recieved target: {parsed_args.target}")

    # Set up parameter overrides
    param_overrides = [
        rclpy.parameter.Parameter(
            "target",
            rclpy.Parameter.Type.STRING,
            parsed_args.target
        )
    ]

    node = JTCClient(parameter_overrides=param_overrides)
    # node.set_parameters(param_overrides)
    
    try:
        rclpy.spin(node)
    except (RuntimeError, SystemExit):
        node.get_logger().info("Shutting down")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
