from ur_asu.custom_libraries.gripperlibraries import EE_pose_to_pushers_2D
from geometry_msgs.msg import PointStamped

class PusherHandler:
    def __init__(self, pub_l, pub_r, node):
        self.pub_l = pub_l
        self.pub_r = pub_r
        self.node = node

    def update(self, ee_position, ee_euler):
        p1, p2 = EE_pose_to_pushers_2D((ee_position, ee_euler))
        now = self.node.get_clock().now().to_msg()
        for pub, pos in zip([self.pub_l, self.pub_r], [p1, p2]):
            msg = PointStamped()
            msg.header.stamp = now
            msg.header.frame_id = "base"
            msg.point.x, msg.point.y, msg.point.z = pos
            pub.publish(msg)
