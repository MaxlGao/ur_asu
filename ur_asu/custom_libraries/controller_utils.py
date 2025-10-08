import rclpy
from rclpy.node import Node

from controller_manager_msgs.srv import SwitchController, ListControllers, LoadController
from builtin_interfaces.msg import Duration

class ControllerManager:
    """Container for all ROS2 controllers. Ensures all needed controllers are loaded and activated/deactivated when needed."""
    def __init__(self, node: Node):
        self.node = node
        self.switch_client = self.node.create_client(SwitchController, '/controller_manager/switch_controller')
        self.list_client = self.node.create_client(ListControllers, '/controller_manager/list_controllers')
        self.load_client = self.node.create_client(LoadController, '/controller_manager/load_controller')

    def list_controllers(self):
        if not self.list_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().error("ListControllers service not available.")
            return []
        future = self.list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self.node, future)
        result = future.result()
        return result.controller if result else []

    def prestart_controllers(self):
        """Ensure all required controllers are loaded and at least one is active."""
        required = [
            self.node.position_controller,
            self.node.velocity_controller,
            self.node.force_controller,
            self.node.passthrough_controller
        ]
        self.node.get_logger().info(f"Loading controllers...")
        controllers = {c.name: c.state for c in self.list_controllers()}

        for ctrl in required:
            if ctrl not in controllers:
                # Try to load missing controller
                self.node.get_logger().warn(f"{ctrl} not loaded. Attempting to load...")
                if not self.load_client.wait_for_service(timeout_sec=2.0):
                    self.node.get_logger().error("LoadController service not available.")
                    continue
                req = LoadController.Request()
                req.name = ctrl
                future = self.load_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                res = future.result()
                if not res or not res.ok:
                    self.node.get_logger().error(f"Failed to load {ctrl}")
                    continue
        self.node.get_logger().info(f"Loading complete.")

        # Start the default position controller to ensure robot can move
        # self.switch_to_controller([self.node.position_controller], [])

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
            self.node.get_logger().info(f"No Controller switch necessary.")
            return True

        self.node.get_logger().info(
            f"Switching controllers..."
        )

        if not self.switch_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().error("Controller switch service not available.")
            return False

        req = SwitchController.Request()
        req.activate_controllers = to_start
        req.deactivate_controllers = to_stop
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        req.timeout = Duration(sec=5)

        future = self.switch_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        result = future.result()

        if not result or not result.ok:
            self.node.get_logger().error(
                f"Controller switch failed. Tried to start={to_start}, stop={to_stop}"
                )
            return False

        self.node.get_logger().info(f"Controller switch success. Now active: {to_start}")
        return True
