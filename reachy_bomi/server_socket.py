#!/usr/bin/env python3

import socket
import threading
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

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
    base_state: float = -1.0


class ServerSocketNode(Node):
    def __init__(self) -> None:
        super().__init__("server_socket_node")
        self.declare_parameter("port", PORT)
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        self.state = ServerState()

        self.linear_vel_pub = self.create_publisher(Float32, "server_socket/linear_vel", 10)
        self.ang_vel_pub = self.create_publisher(Float32, "server_socket/angular_vel", 10)
        self.base_state_pub = self.create_publisher(Float32, "server_socket/base_state", 10)

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

        if "nine region" in msg:
            self.state.base_state = 1.0
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