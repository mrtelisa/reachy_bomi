#!/usr/bin/env python3

import os
import signal
import socket
import subprocess
import threading
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from reachy_bomi.scenarios import SCENARIO_NAMES

PORT = 5051
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"

# Fallback bind IP — replace with this machine's actual IP on your network.
# Used only if get_local_ip() can't determine one (e.g. no default route,
# such as two machines wired directly with no gateway configured).
DEFAULT_BIND_IP = "192.168.1.100"


def get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 53))
        return sock.getsockname()[0]
    except OSError:
        return DEFAULT_BIND_IP
    finally:
        sock.close()


@dataclass
class ServerState:
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    base_state: float = -1.0


class ServerSocketNode(Node):
    def __init__(self) -> None:
        super().__init__("socket_server_node")
        self.declare_parameter("port", PORT)
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        self.state = ServerState()

        self.linear_vel_pub = self.create_publisher(Float32, "socket_server/linear_vel", 10)
        self.ang_vel_pub = self.create_publisher(Float32, "socket_server/angular_vel", 10)
        self.base_state_pub = self.create_publisher(Float32, "socket_server/base_state", 10)

        self._scenario_proc = None

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
            # This is made to allow that different clients can connect to the server at the 
            # same time, each one in a different thread
            thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
            thread.start()

    def _handle_client(self, conn: socket.socket) -> None:
        connected = True
        # Buffer across recv() calls: TCP is a byte stream, a single recv()
        # can contain multiple '\n'-delimited messages, a partial one, or both.
        buffer = ""
        try:
            while connected and rclpy.ok():
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data.decode(FORMAT)
                while "\n" in buffer:
                    msg, buffer = buffer.split("\n", 1)
                    msg = msg.strip()
                    if not msg:
                        continue
                    self._decode_msg(msg)
                    if msg == DISCONNECT_MESSAGE:
                        connected = False
        finally:
            conn.close()
            # Whatever the reason the client went away (clean disconnect, closed
            # terminal, dropped connection), stop the scenario it had launched.
            self._terminate_scenario_process()

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

        if "scenario:" in msg:
            self._launch_scenario(msg)

    @staticmethod
    def _extract_float(message: str, key: str) -> float:
        start = message.find(key)
        if start < 0:
            return 0.0
        payload = message[start + len(key):].split()[0]
        return float(payload)

    @staticmethod
    def _extract_field(message: str, key: str, default: str) -> str:
        for token in message.split():
            if token.startswith(f"{key}:"):
                return token[len(key) + 1:]
        return default

    def _launch_scenario(self, msg: str) -> None:
        scenario = self._extract_field(msg, "scenario", "")
        if scenario not in SCENARIO_NAMES:
            self.get_logger().warning(f"Unknown scenario '{scenario}', ignoring")
            return

        start_rviz = self._extract_field(msg, "rviz", "true")
        record = self._extract_field(msg, "record", "true")

        self._terminate_scenario_process()

        cmd = [
            "ros2", "launch", "reachy_bomi", "bomi_control.launch.py",
            f"scenario:={scenario}",
            f"start_rviz:={start_rviz}",
            f"record:={record}",
        ]
        self.get_logger().info(f"Launching scenario '{scenario}' (rviz={start_rviz}, record={record})")
        self._scenario_proc = subprocess.Popen(cmd, start_new_session=True)

    def _terminate_scenario_process(self) -> None:
        if self._scenario_proc is None or self._scenario_proc.poll() is not None:
            return
        self.get_logger().info("Stopping the previous scenario before launching the new one")
        pgid = os.getpgid(self._scenario_proc.pid)
        os.killpg(pgid, signal.SIGINT)
        try:
            self._scenario_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self._scenario_proc.wait()


def main() -> None:
    rclpy.init()
    node = ServerSocketNode()
    try:
        rclpy.spin(node)
    finally:
        node._terminate_scenario_process()
        node.server.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()