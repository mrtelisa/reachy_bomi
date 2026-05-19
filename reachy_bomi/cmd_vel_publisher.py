#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
#from nav2_msgs.action import NavigateToPose
import math

# Map limits (x_min, x_max, y_min, y_max) per world, from world file bounding boxes
WORLD_LIMITS = {
    "familiarization": (-10.1,  10.15, -7.15,  6.2),
    "labyrinth1":      (-1.12,  21.13, -2.23,  6.84),
    "labyrinth2":      (-1.31,  19.94, -2.29,  5.97),
    "mid_tests":       (-10.4,  10.43, -6.68,  5.68),
    "final_test":      (-4.05,  20.95, -1.56,  15.64),
}

class ServerData(Node):
    """
    This class is used to share variables between functions
    """
    def __init__(self):
        super().__init__('reachy_base_controller')

        # --- State variables ---
        self.reachy_position = Point()
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.x_coordinate = 0.0
        self.y_coordinate = 0.0
        self.x_coordinate_arrived = False
        self.y_coordinate_arrived = False
        self.base_state = -1.0
        self.send_coordinates = False
        self.map_name = None
        self.already_acquired_map_name = False
        

        # Variable to store the information about the first goal
        # Used to understand if it is necessary to cancel the goal
        self.first_goal = True
        
        # Compute map region
        self.x_min= None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        # --- PUBLISHERS ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- SUBSCRIBERS ---
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_clbk, 10)
        self.map_name_sub = self.create_subscription(Float32, 'server_socket/map_name', self.map_name_clbk, 10)
        self.base_state_sub = self.create_subscription(Float32, 'server_socket/base_state', self.base_state_clbk, 10)
        self.linear_vel_sub = self.create_subscription(Float32, 'server_socket/linear_vel', self.linear_vel_clbk, 10)
        self.ang_vel_sub = self.create_subscription(Float32, 'server_socket/angular_vel', self.angular_vel_clbk, 10)
        self.x_coor_sub = self.create_subscription(Float32, 'server_socket/x_coordinate', self.x_coordinate_clbk, 10)
        self.y_coor_sub = self.create_subscription(Float32, 'server_socket/y_coordinate', self.y_coordinate_clbk, 10)

        # --- ACTION CLIENT (Nav2) ---
        '''try:
            self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        except ImportError:
            self.get_logger().error("Nav2 messages not found. Navigation mode will be disabled.")'''

        # --- TIMER LOOP (10Hz) ---
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Reachy 2 Base Controller Started")

    # --- CALLBACKS ---
    def odom_clbk(self, msg):
        """
        Simple odom callback that update Reachy position into a global variable
        :param 
            msg: the odom msg from subcriber
        """
        self.reachy_position = msg.pose.pose.position
        #print("Reachy position: x: " + str(reachy_position.x) + " y: " + str(reachy_position.y))


    def map_name_clbk(self, msg):
        """
        Once the name of the map is received, this function stores the values of the map
        """
        if not self.already_acquired_map_name:
            if msg.data == 1.0:                         # familiarization
                limits = WORLD_LIMITS["familiarization"]
            elif msg.data in (2.0, 3.0, 4.0, 5.0):     # train1-4 (labyrinth1)
                limits = WORLD_LIMITS["labyrinth1"]
            elif msg.data in (6.0, 11.0):               # test1, test2 (mid_tests)
                limits = WORLD_LIMITS["mid_tests"]
            elif msg.data in (7.0, 8.0, 9.0, 10.0):    # train5-8 (labyrinth2)
                limits = WORLD_LIMITS["labyrinth2"]
            elif msg.data == 12.0:                      # final_test
                limits = WORLD_LIMITS["final_test"]
            else:
                limits = None
            if limits is not None:
                self.x_min, self.x_max, self.y_min, self.y_max = limits
            self.already_acquired_map_name = True
            self.get_logger().info(f"Map Limits Acquired: X[{self.x_min}:{self.x_max}]")

    def base_state_clbk(self, msg): 
        self.base_state = msg.data
        #print("Base State : " + str(msg.data))

    def linear_vel_clbk(self, msg): 
        self.linear_vel = msg.data

    def angular_vel_clbk(self, msg): 
        self.angular_vel = msg.data
    
    def x_coordinate_clbk(self, msg):
        self.x_coordinate = msg.data
        self.x_coordinate_arrived = True
        print("X coordinate arrived: " + str(self.x_coordinate))

    def y_coordinate_clbk(self, msg):
        self.y_coordinate = msg.data
        self.y_coordinate_arrived = True
        print("Y coordinate: " + str(self.y_coordinate))

    # --- LOGIC ---
    def check_valid_coordinate(self, x, y):
        """
        Simple function that computes if the selected coordinates are valid or not
        Of course they depend on Reachy position, since taret = Reachy + selected target
        :params
            x_coor: the x coordinate of the target
            y_coor: the y coordinate of the target
        :return
            [bool]: True if the coordinates are valid, False otherwise
        """
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def control_loop(self):
        # Mode 1: nine regions GUI -> move Reachy using /cmd_vel msgs
        if self.base_state == 1.0:
            vel_msg = Twist()
            vel_msg.linear.x = self.linear_vel
            vel_msg.angular.z = self.angular_vel
            self.cmd_vel_pub.publish(vel_msg)

            print("I have published Lin Vel: " + str(self.linear_vel) + " Angular Vel: " + str(self.angular_vel))

        # Mode 0: if odom GUI is stated
        elif self.base_state == 0.0:
            if self.x_coordinate_arrived and self.y_coordinate_arrived:
                #Restore the values for the next turn
                self.x_coordinate_arrived = False
                self.y_coordinate_arrived = False
                self.send_nav2_goal()

    '''def send_nav2_goal(self):
        # Declare the target position
        target = Point()
        target.x = self.x_coordinate + self.reachy_position.x
        target.y = self.y_coordinate + self.reachy_position.y

        if not self.check_valid_coordinate(target.x, target.y):
            self.get_logger().warn("Target out of map! Clipping coordinates...")
            target.x = max(min(target.x, self.x_max - 0.5), self.x_min + 0.5)
            target.y = max(min(target.y, self.y_max - 0.5), self.y_min + 0.5)
            
        self.get_logger().info(f"Sending Goal: X={target.x}, Y={target.y}")

        # Goal creation for Nav2
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = target.x
        goal_msg.pose.pose.position.y = target.y
        goal_msg.pose.pose.orientation.w = 1.0

        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal_msg)'''

def main(args=None):
    rclpy.init(args=args)
    node = ServerData()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()