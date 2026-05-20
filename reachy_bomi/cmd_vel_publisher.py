#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

# Map limits (x_min, x_max, y_min, y_max) per world, from world file bounding boxes
WORLD_LIMITS = {
    "familiarization": (-10.1,  10.15, -7.15,  6.2),
    "labyrinth1":      (-1.12,  21.13, -2.23,  6.84),
    "labyrinth2":      (-1.31,  19.94, -2.29,  5.97),
    "mid_tests":       (-10.4,  10.43, -6.68,  5.68),
    "final_test":      (-4.05,  20.95, -1.56,  15.64),
}

_MAP_ID_TO_WORLD = {
    1.0:  "familiarization",
    2.0:  "labyrinth1", 3.0: "labyrinth1", 4.0: "labyrinth1", 5.0: "labyrinth1",
    6.0:  "mid_tests",  11.0: "mid_tests",
    7.0:  "labyrinth2", 8.0: "labyrinth2", 9.0: "labyrinth2", 10.0: "labyrinth2",
    12.0: "final_test",
}


class CmdVelPublisher(Node):
    def __init__(self) -> None:
        super().__init__("reachy_base_controller")

        self.reachy_position = Point()
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.x_coordinate = 0.0
        self.y_coordinate = 0.0
        self.x_coordinate_arrived = False
        self.y_coordinate_arrived = False
        self.base_state = -1.0
        self.already_acquired_map_name = False
        self.x_min = self.x_max = self.y_min = self.y_max = None

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(Float32, "server_socket/map_name", self._map_name_cb, 10)
        self.create_subscription(Float32, "server_socket/base_state", self._base_state_cb, 10)
        self.create_subscription(Float32, "server_socket/linear_vel", self._linear_vel_cb, 10)
        self.create_subscription(Float32, "server_socket/angular_vel", self._ang_vel_cb, 10)
        self.create_subscription(Float32, "server_socket/x_coordinate", self._x_coor_cb, 10)
        self.create_subscription(Float32, "server_socket/y_coordinate", self._y_coor_cb, 10)

        self.timer = self.create_timer(0.1, self._control_loop)
        self.get_logger().info("Reachy base controller started")

    def _odom_cb(self, msg: Odometry) -> None:
        self.reachy_position = msg.pose.pose.position

    def _map_name_cb(self, msg: Float32) -> None:
        if self.already_acquired_map_name:
            return
        world_name = _MAP_ID_TO_WORLD.get(msg.data)
        if world_name and world_name in WORLD_LIMITS:
            self.x_min, self.x_max, self.y_min, self.y_max = WORLD_LIMITS[world_name]
            self.get_logger().info(f"Map limits set for '{world_name}'")
        self.already_acquired_map_name = True

    def _base_state_cb(self, msg: Float32) -> None:
        self.base_state = msg.data

    def _linear_vel_cb(self, msg: Float32) -> None:
        self.linear_vel = msg.data

    def _ang_vel_cb(self, msg: Float32) -> None:
        self.angular_vel = msg.data

    def _x_coor_cb(self, msg: Float32) -> None:
        self.x_coordinate = msg.data
        self.x_coordinate_arrived = True

    def _y_coor_cb(self, msg: Float32) -> None:
        self.y_coordinate = msg.data
        self.y_coordinate_arrived = True

    def _check_valid_coordinate(self, x: float, y: float) -> bool:
        if None in (self.x_min, self.x_max, self.y_min, self.y_max):
            return True
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def _control_loop(self) -> None:
        if self.base_state == 1.0:
            msg = Twist()
            msg.linear.x = self.linear_vel
            msg.angular.z = self.angular_vel
            self.cmd_vel_pub.publish(msg)

        elif self.base_state == 0.0 and self.x_coordinate_arrived and self.y_coordinate_arrived:
            self.x_coordinate_arrived = False
            self.y_coordinate_arrived = False
            self.get_logger().warn(
                "Odom/Nav2 goal mode requested but Nav2 is not active in this setup"
            )


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
