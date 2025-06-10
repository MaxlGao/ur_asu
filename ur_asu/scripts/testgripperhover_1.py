import time
import rclpy
import numpy as np
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from builtin_interfaces.msg import Duration
from action_msgs.msg import GoalStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance

from ur_asu.custom_libraries.actionlibraries import move  # <-- your function here

# Script makes a list of trajectories moving the EE up and down while changing gripper width
# as to hopefully make the gripper tips keep at a constant height. 
# First attempt on this type of task

def generate_gripper_commands(step):
    # Create ascending and descending sequences
    up = list(range(0, 1000 + step, step))       # 0 to 1000
    down = list(range(1000 - step, -step, -step)) # 800 to 0
    full_seq = up + down

    return {
        f"traj{i+1}": str(width) for i, width in enumerate(full_seq)
    }

GRIPPER_COMMANDS = generate_gripper_commands(step=100)

GRIPPER_TABLE = {
       0: 0.153,
     200: 0.150,
     400: 0.147,
     600: 0.141,
     800: 0.132,
    1000: 0.114,
}

def gripper_interpolate(gripper_width):
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

class JTCClient(Node):
    def __init__(self):
        super().__init__("trajectory_executor")
        self.declare_parameter("controller_name", "scaled_joint_trajectory_controller")
        self.declare_parameter("joints", [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"])

        controller_name = self.get_parameter("controller_name").value + "/follow_joint_trajectory"
        self.joints = self.get_parameter("joints").value
        self._action_client = ActionClient(self, FollowJointTrajectory, controller_name)
        self._gripper_pub = self.create_publisher(String, "/gripper_command", 10)

        self.get_logger().info(f"Waiting for action server on {controller_name}")
        self._action_client.wait_for_server()

        # Define poses
        base_orientation = [0, 180, 0]

        start_duration = 3
        segment_duration = 1 # Don't go below this number
        vertical_offset = 0.050

        self.trajectories = {
            "traj0": move([0.09, -0.549, GRIPPER_TABLE[0] + vertical_offset], base_orientation, start_duration)
        }
        for traj_name, width_str in GRIPPER_COMMANDS.items():
            height = gripper_interpolate(float(width_str)) + vertical_offset
            self.trajectories[traj_name] = move([0.09, -0.549, height], base_orientation, segment_duration)

        self.goals = self.parse_trajectories()
        self.i = 0
        self.execute_next_trajectory()

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
        self.get_logger().info(f"Sending gripper command: {cmd}")
        self._gripper_pub.publish(msg)

    def execute_next_trajectory(self):
        if self.i >= len(self.goals):
            self.get_logger().info("Done with all trajectories")
            raise SystemExit

        traj_name = list(self.goals)[self.i]
        self.i += 1

        if traj_name in GRIPPER_COMMANDS:
            # Do gripper before moving
            self.publish_gripper(GRIPPER_COMMANDS[traj_name])

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
