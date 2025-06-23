import time
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
import numpy as np
import argparse

from ur_asu.custom_libraries.actionlibrariesmax import move, hover_over

# Script allows for gripper positions to be sent as inputs. 
GRIPPER_FINGER_OFFSET = 0.0058 # m

GRIPPER_TABLE = { # Known, measured values. Gripper width in 0.1mm.
       0: 0.153,
     200: 0.150,
     400: 0.147,
     600: 0.141,
     800: 0.132,
    1000: 0.114,
}

# Virtual table level in mm
VERTICAL_OFFSET = 0.003 # 0.000 = gripper tips always on table (dangerous)

def gripper_width_to_height(gripper_width):
    # Clamp input to valid range
    if gripper_width <= 0:
        return GRIPPER_TABLE[0]
    if gripper_width >= 1000:
        return GRIPPER_TABLE[1000]

    # Sort the keys to find where gripper_width fits
    keys = sorted(GRIPPER_TABLE.keys())
    for i in range(len(keys) - 1):
        low = keys[i]
        high = keys[i + 1]
        if low <= gripper_width <= high:
            # Linear interpolation
            low_val = GRIPPER_TABLE[low]
            high_val = GRIPPER_TABLE[high]
            t = (gripper_width - low) / (high - low)
            return low_val + t * (high_val - low_val)

def height_to_gripper_width(height):
    # Convert GRIPPER_TABLE to (height -> width) tuples for inverse lookup
    items = sorted(((v, k) for k, v in GRIPPER_TABLE.items()), reverse=True)

    # Clamp to valid range
    if height >= items[0][0]:
        return items[0][1]
    if height <= items[-1][0]:
        return items[-1][1]

    # Find surrounding interval
    for i in range(len(items) - 1):
        high_val, low_key = items[i]
        low_val, high_key = items[i + 1]
        if high_val >= height >= low_val:
            t = (height - low_val) / (high_val - low_val)
            return int(high_key + t * (low_key - high_key))

def EE_pose_to_pushers_2D(EE_pose):
    """
    Returns the gripper positions given the current EE pose, assuming the gripper width is set according to height.
    Currently doesn't intelligently distinguish which gripper's which.
    Pusher 1 is the left one when looking down the camera.
    Pusher positions are [x, y, 0]
    """
    position, orientation = EE_pose
    center_x, center_y, height = position
    height -= VERTICAL_OFFSET
    yaw = orientation[2]
    yaw_r = np.deg2rad(yaw)
    gripper_width = height_to_gripper_width(height) * 0.0001 # Conversion to meters
    half_width = gripper_width / 2

    dx = half_width * np.cos(yaw_r)
    dy = half_width * np.sin(yaw_r)
    pusher1_position = [center_x + dx, center_y + dy, 0.0]
    pusher2_position = [center_x - dx, center_y - dy, 0.0]

    return pusher1_position, pusher2_position

def pushers_to_EE_pose_2D(pusher_1_position, pusher_2_position, given_yaw=0):
    """
    Returns the EE pose such that the gripper can place its two tips at the given XY locations
    pusher_positions are [x1, y1, z1], [x2, y2, z2] meters, where zs are ignored
    returns position, orientation_deg
    Prone to singularity (pushers at the same point)
    """
    pos1_np = np.array(pusher_1_position)
    pos2_np = np.array(pusher_2_position)
    gripper_width = np.linalg.norm(pos1_np-pos2_np)
    # Assert that the two points are within the gripper maximum (110 mm in this case)
    if gripper_width > 0.11:
        return None
    gripper_height = gripper_width_to_height(gripper_width*10000) # conversion to their weird measurement
    center = (pos1_np + pos2_np) / 2
    position = [center[0], center[1], gripper_height + VERTICAL_OFFSET]
    if gripper_width < 0.01:
        yaw = given_yaw
    else:
        # If gripper width is more than 10 mm, probably okay to measure angle.
        yaw = np.atan2(pos1_np[1]-pos2_np[1], pos1_np[0]-pos2_np[0])
        yaw = np.rad2deg(yaw)
        print(f"YAW IS {yaw}")
    orientation = [0, 180, yaw]
    return position, orientation

def canonicalize_euler(orientation):
    """Forces euler angles near the form (-180, 0, yaw') to take the equivalent form (0, 180, yaw)"""
    roll, pitch, yaw = orientation
    if abs(pitch) < 1 and abs(abs(roll) - 180) < 1:
        return (0.0, 180.0, (yaw % 360)-180)
    else:
        return orientation

def pointspan_to_pushers(pusher_1, span, yaw_d):
    """Provides a pair of pusher positions given pusher 1's position, span, and the angle"""
    yaw_r = np.deg2rad(yaw_d)
    dx = span * np.cos(yaw_r)
    dy = span * np.sin(yaw_r)
    new_x = pusher_1[0] - dx
    new_y = pusher_1[1] - dy
    pusher_2 = [new_x, new_y, 0.0]
    return pusher_1, pusher_2

def pointspan_to_EE_pose_2D(pusher_1, span, yaw_d):
    """
    Provides a EE pose given pusher 1's position, span, and the angle
    Pusher positions are [x, y, --] meters
    given yaw is in degrees
    span is in meters
    pose is [position, euler orientation degrees]
    """
    yaw_r = np.deg2rad(yaw_d)
    half_span = span/2
    dx = half_span * np.cos(yaw_r)
    dy = half_span * np.sin(yaw_r)
    cen_x = pusher_1[0] - dx
    cen_y = pusher_1[1] - dy
    gripper_height = gripper_width_to_height(span*10000) # conversion to their weird measurement
    position = [cen_x, cen_y, gripper_height + VERTICAL_OFFSET]
    orientation = [0, 180, yaw_d]
    return position, orientation

def EE_pose_to_pointspan(EE_pose):
    """
    Returns the pointspan given the current EE pose, assuming the gripper width is set according to height.
    Pusher 1 is the left one when looking down the camera.
    Pusher positions are [x, y, 0]
    """
    position, orientation = EE_pose
    center_x, center_y, height = position
    height -= VERTICAL_OFFSET
    yaw_d = orientation[2]
    yaw_r = np.deg2rad(yaw_d)
    gripper_width = height_to_gripper_width(height) * 0.0001 # Conversion to meters
    half_width = gripper_width / 2

    dx = half_width * np.cos(yaw_r)
    dy = half_width * np.sin(yaw_r)
    pusher1_position = [center_x + dx, center_y + dy, 0.0]

    return pusher1_position, gripper_width, yaw_d


def pose_text(pose):
    position, euler = pose
    return f"""XYZ: {1000*position[0]:.1f}, {1000*position[1]:.1f}, {1000*position[2]:.1f} mm
RPY: {euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f} deg"""

def pushers_text(pusher_1_pos, pusher_2_pos):
    return f"""
    Pusher 1: ({1000*pusher_1_pos[0]:.1f}, {1000*pusher_1_pos[1]:.1f}, {1000*pusher_1_pos[2]:.1f}) mm
    Pusher 2: ({1000*pusher_2_pos[0]:.1f}, {1000*pusher_2_pos[1]:.1f}, {1000*pusher_2_pos[2]:.1f}) mm"""

def pointspan_to_trajectories(EE_pose_now, pusher_1_target, span_target, yaw_target, duration):
    """
    Provides a list of joint angle trajectories based on where you want your pushers
    Pusher positions are [x, y, --] meters
    given yaw is in degrees
    span is in meters
    pose is [position, euler orientation degrees]
    """
    print(f"""We're currently at:
{pose_text(EE_pose_now)}""")
    pusher_1_now, pusher_2_now = EE_pose_to_pushers_2D(EE_pose_now)
    print("pushers now")
    print(pushers_text(pusher_1_now, pusher_2_now))
    pos1_now_np = np.array(pusher_1_now)
    pos2_now_np = np.array(pusher_2_now)
    pusher_1_target, pusher_2_target = pointspan_to_pushers(pusher_1_target, span_target, yaw_target)
    print("pushers target")
    print(pushers_text(pusher_1_target, pusher_2_target))
    pos1_target_np = np.array(pusher_1_target)
    pos2_target_np = np.array(pusher_2_target)
    width_now = np.linalg.norm(pos1_now_np - pos2_now_np)
    width_target = np.linalg.norm(pos1_target_np - pos2_target_np)
    # Checks if the 40mm width mark is crossed
    if width_now <= 0.04 <= width_target or width_target <= 0.04 <= width_now:
        # If it is, we need an intermediate waypoint. 
        # Otherwise, the controller will increase/decrease height linearly, which can bump the gripper into the table.
        # This is to have the controller raise/lower more slowly when it matters most.

        # If pushers move linearly at constant speed, then width will increase linearly
        # We need to find at what point does width = 0.04.
        proportion = abs(0.04 - width_now) / abs(width_target - width_now)
        # Then, find the vector each pusher travels along and find the appropriate point.
        vec1 = pos1_target_np - pos1_now_np
        vec2 = pos2_target_np - pos2_now_np
        prop_vec1 = proportion * vec1
        prop_vec2 = proportion * vec2
        intermed_pos1 = pos1_now_np + prop_vec1
        intermed_pos2 = pos2_now_np + prop_vec2
        print("pushers intermed")
        print(pushers_text(intermed_pos1, intermed_pos2))
        intermed_position, intermed_rpy = pushers_to_EE_pose_2D(intermed_pos1, intermed_pos2)
        position, rpy = pointspan_to_EE_pose_2D(pusher_1_target, span_target, yaw_target)
        print(f"""Got coordinates: 
{pose_text([intermed_position, intermed_rpy])}
           then:
{pose_text([position, rpy])}""")
        return [move(intermed_position, intermed_rpy, duration), move(position, rpy, duration)]
    else:
        position, rpy = pointspan_to_EE_pose_2D(pusher_1_target, span_target, yaw_target)
        print(f"""Got coordinates: 
{pose_text([position, rpy])}""")
        return [move(position, rpy, duration)]

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
