#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class CmdVelPublisher(Node):
    def __init__(self) -> None:
        super().__init__("reachy_base_controller")

        self.linear_vel = 0.0
        self.angular_vel = 0.0
        # Defaults to velocity mode: this deployment only ever uses cmd_vel
        # teleop, so there's no need to wait for the "nine region" handshake
        # from socket_server before /cmd_vel starts being published.
        self.base_state = 1.0

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_subscription(Float32, "socket_server/base_state", self._base_state_cb, 10)
        self.create_subscription(Float32, "socket_server/linear_vel", self._linear_vel_cb, 10)
        self.create_subscription(Float32, "socket_server/angular_vel", self._ang_vel_cb, 10)

        # Matches SEND_HZ in socket_client.py so /cmd_vel republishes at the
        # same rate the operator PC actually computes velocities
        self.timer = self.create_timer(1.0 / 20, self._control_loop)
        self.get_logger().info("Reachy base controller started")

    def _base_state_cb(self, msg: Float32) -> None:
        self.base_state = msg.data

    def _linear_vel_cb(self, msg: Float32) -> None:
        self.linear_vel = msg.data

    def _ang_vel_cb(self, msg: Float32) -> None:
        self.angular_vel = msg.data

    def _control_loop(self) -> None:
        # Velocity mode: republish hand-derived velocities to the mobile base
        if self.base_state == 1.0:
            msg = Twist()
            msg.linear.x = self.linear_vel
            msg.angular.z = self.angular_vel
            self.cmd_vel_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()