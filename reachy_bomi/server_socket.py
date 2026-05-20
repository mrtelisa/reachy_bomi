#!/usr/bin/env python3

import socket
import threading
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32

from reachy_bomi.scenarios import SCENARIO_IDS

HEADER = 64
PORT = 5051
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"


def get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 53))
        return sock.getsockname()[0]
    finally:
        sock.close()


@dataclass
class ServerState:
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    x_coordinate: float = 0.0
    y_coordinate: float = 0.0
    map_name: float = 0.0
    base_state: float = -1.0
    vector_angle: int = 0
    vector_amplitude: int = 0


class ServerSocketNode(Node):
    def __init__(self) -> None:
        super().__init__("server_socket_node")
        self.declare_parameter("port", PORT)
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        self.state = ServerState()

        self.linear_vel_pub = self.create_publisher(Float32, "server_socket/linear_vel", 10)
        self.ang_vel_pub = self.create_publisher(Float32, "server_socket/angular_vel", 10)
        self.x_coor_pub = self.create_publisher(Float32, "server_socket/x_coordinate", 10)
        self.y_coor_pub = self.create_publisher(Float32, "server_socket/y_coordinate", 10)
        self.map_name_pub = self.create_publisher(Float32, "server_socket/map_name", 10)
        self.base_state_pub = self.create_publisher(Float32, "server_socket/base_state", 10)
        self.amplitude_vector_pub = self.create_publisher(Int32, "server_socket/vector_amplitude", 10)
        self.angle_vector_pub = self.create_publisher(Int32, "server_socket/vector_angle", 10)

        local_ip = get_local_ip()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((local_ip, self.port))
        self.server.listen()
        self.get_logger().info(f"Socket server listening on {local_ip}:{self.port}")

        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()

    def _accept_loop(self) -> None:
        while rclpy.ok():
            conn, addr = self.server.accept()
            self.get_logger().info(f"New connection from {addr}")
            thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
            thread.start()

    def _handle_client(self, conn: socket.socket) -> None:
        connected = True
        while connected and rclpy.ok():
            data = conn.recv(1024)
            if not data:
                break
            msg = data.decode(FORMAT).strip()
            self._decode_msg(msg)
            if msg == DISCONNECT_MESSAGE:
                connected = False
        conn.close()

    def _decode_msg(self, msg: str) -> None:
        if "lin_vel:" in msg:
            self.state.linear_vel = self._extract_float(msg, "lin_vel:")
            self.linear_vel_pub.publish(Float32(data=self.state.linear_vel))

        if "ang_vel:" in msg:
            self.state.angular_vel = self._extract_float(msg, "ang_vel:")
            self.ang_vel_pub.publish(Float32(data=self.state.angular_vel))

        if "x:" in msg:
            self.state.x_coordinate = self._extract_float(msg, "x:")
            self.x_coor_pub.publish(Float32(data=self.state.x_coordinate))

        if "y:" in msg:
            self.state.y_coordinate = self._extract_float(msg, "y:")
            self.y_coor_pub.publish(Float32(data=self.state.y_coordinate))

        if "angle" in msg and "amplitude" in msg:
            parts = msg.split()
            try:
                angle_idx = parts.index("angle")
                amp_idx = parts.index("amplitude")
                self.state.vector_angle = int(parts[angle_idx + 1])
                self.state.vector_amplitude = int(parts[amp_idx + 1])
                self.angle_vector_pub.publish(Int32(data=self.state.vector_angle))
                self.amplitude_vector_pub.publish(Int32(data=self.state.vector_amplitude))
            except (ValueError, IndexError):
                self.get_logger().warning(f"Cannot decode vector message: {msg}")

        for scenario_name, scenario_id in SCENARIO_IDS.items():
            if scenario_name in msg:
                self.state.map_name = float(scenario_id)
                self.map_name_pub.publish(Float32(data=self.state.map_name))
                self.get_logger().info(f"Scenario: {scenario_name}")

        if "nine region" in msg:
            self.state.base_state = 1.0
            self.base_state_pub.publish(Float32(data=self.state.base_state))
        elif "odom" in msg:
            self.state.base_state = 0.0
            self.base_state_pub.publish(Float32(data=self.state.base_state))

    @staticmethod
    def _extract_float(message: str, key: str) -> float:
        start = message.find(key)
        if start < 0:
            return 0.0
        payload = message[start + len(key):].split()[0]
        return float(payload)


def main() -> None:
    rclpy.init()
    node = ServerSocketNode()
    try:
        rclpy.spin(node)
    finally:
        node.server.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
