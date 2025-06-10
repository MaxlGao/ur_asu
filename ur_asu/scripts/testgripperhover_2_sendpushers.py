import time
import rclpy
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from builtin_interfaces.msg import Duration
from action_msgs.msg import GoalStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, PointStamped
from control_msgs.msg import JointTolerance

from ur_asu.custom_libraries.actionlibraries import move  # <-- your function here
from ur_asu.custom_libraries.actionlibrariesmax import spin_around  # <-- your function here

GRIPPER_TABLE = { # Known, measured values. Gripper width in 0.1mm.
       0: 0.153,
     200: 0.150,
     400: 0.147,
     600: 0.141,
     800: 0.132,
    1000: 0.114,
}

# Virtual table level in mm
VERTICAL_OFFSET = 0.005 # 0.000 = gripper tips always on table (dangerous)

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
    returns position, orientation
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
    orientation = [0, 180, yaw]
    return position, orientation

def canonicalize_euler(orientation):
    """Forces euler angles near the form (-180, 0, yaw') to take the equivalent form (0, 180, yaw)"""
    roll, pitch, yaw = orientation
    if abs(pitch) < 1 and abs(abs(roll) - 180) < 1:
        return (0.0, 180.0, (yaw % 360)-180)

def send_pushers(pusher_1_now, pusher_2_now, pusher_1_target, pusher_2_target, duration):
    """
    Provides a list of joint angle trajectories based on where you want your pushers
    Pusher positions are [x, y, --] meters
    """
    pos1_now_np = np.array(pusher_1_now)
    pos2_now_np = np.array(pusher_2_now)
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
        intermed_position, intermed_rpy = pushers_to_EE_pose_2D(intermed_pos1, intermed_pos2)
        position, rpy = pushers_to_EE_pose_2D(pusher_1_target, pusher_2_target)
        print(f"""Got coordinates: {1000*intermed_position[0]:.1f}, {1000*intermed_position[2]:.1f}, {1000*intermed_position[2]:.1f} mm
            then {1000*position[0]:.1f}, {1000*position[1]:.1f}, {1000*position[2]:.1f}mm""")
        return [move(intermed_position, intermed_rpy, duration), move(position, rpy, duration)]
    else:
        position, rpy = pushers_to_EE_pose_2D(pusher_1_target, pusher_2_target)
        return [move(position, rpy, duration)]
    
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
    def __init__(self):
        super().__init__("trajectory_executor")
        self.declare_parameter("controller_name", "scaled_joint_trajectory_controller")
        self.declare_parameter("joints", [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"])


        self.ee_position = np.array([-0.144, -0.435, 0.202])
        self.ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.ee_euler = np.array([0.0, 0.0, 0.0])
        controller_name = self.get_parameter("controller_name").value + "/follow_joint_trajectory"
        self.joints = self.get_parameter("joints").value
        self._action_client = ActionClient(self, FollowJointTrajectory, controller_name)
        self._gripper_pub = self.create_publisher(String, "/gripper_command", 10)
        self.subscription = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            10)
        self.pusher_pub_l = self.create_publisher(PointStamped, "/gripper_pusher_left", 10)
        self.pusher_pub_r = self.create_publisher(PointStamped, "/gripper_pusher_right", 10)

        self.get_logger().info(f"Waiting for action server on {controller_name}")
        self._action_client.wait_for_server()

        # Define poses
        base_orientation = [0, 180, 0]

        start_duration = 3
        segment_duration = 2 # 1.5 is a mite aggressive. 

        start_position = [0.00, -0.550, GRIPPER_TABLE[0] + VERTICAL_OFFSET]
        first_move = move(start_position, base_orientation, start_duration)
        start_pusher1, start_pusher2 = EE_pose_to_pushers_2D((start_position, base_orientation))
        self.trajectories = {}
        self.trajectories = append_new_traj(self.trajectories, first_move)
        
        pusher_coordinates = [[[0.03, -0.52, 0.0], [-0.03, -0.52, 0.0]],
                              [[0.03, -0.52, 0.0], [-0.03, -0.58, 0.0]],
                              [[0.03, -0.52, 0.0], [+0.03, -0.58, 0.0]],
                              [[0.03, -0.52, 0.0], [+0.03, -0.52, 0.0]]]
        for idx in range(len(pusher_coordinates)+1):
            if idx == 0:
                coord = pusher_coordinates[idx]
                moves = send_pushers(start_pusher1, start_pusher2, coord[0], coord[1], segment_duration)
            elif idx == len(pusher_coordinates):
                coord = pusher_coordinates[idx-1]
                moves = send_pushers(coord[0], coord[1], start_pusher1, start_pusher2, segment_duration)
            else:
                prev_coord = pusher_coordinates[idx-1]
                next_coord = pusher_coordinates[idx]
                moves = send_pushers(prev_coord[0], prev_coord[1], next_coord[0], next_coord[1], segment_duration)
            for mov in moves:
                self.trajectories = append_new_traj(self.trajectories, mov)

        self.goals = self.parse_trajectories()
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
        print(f"Pusher 1: ({1000*pusher_1_pos[0]:.1f}, {1000*pusher_1_pos[1]:.1f}, {1000*pusher_1_pos[2]:.1f}) mm")
        print(f"Pusher 2: ({1000*pusher_2_pos[0]:.1f}, {1000*pusher_2_pos[1]:.1f}, {1000*pusher_2_pos[2]:.1f}) mm")

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
            self.get_logger().info("Done with all trajectories")
            raise SystemExit

        traj_name = list(self.goals)[self.i]
        self.i += 1

        self.execute_trajectory(traj_name)


    def execute_trajectory(self, traj_name):
        self.get_logger().info(f"▶ Executing trajectory {traj_name}")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self.goals[traj_name]
        goal.goal_time_tolerance = Duration(sec=0, nanosec=500_000_000)
        goal.goal_tolerance = [JointTolerance(position=0.01, velocity=0.01, name=name) for name in self.joints]

        self._send_goal_future = self._action_client.send_goal_async(goal)
        self._send_goal_future.add_done_callback(lambda f: self.goal_response_callback(f, traj_name))

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
        self.get_logger().info(f"✔ Trajectory {traj_name} completed with status: {self.status_to_str(status)}")

        if status == GoalStatus.STATUS_SUCCEEDED:
            time.sleep(1)
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
    node = JTCClient()
    try:
        rclpy.spin(node)
    except (RuntimeError, SystemExit):
        node.get_logger().info("Shutting down")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
