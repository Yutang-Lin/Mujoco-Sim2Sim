import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Change according to your message type
import time
from crc import CRC

import numpy as np
from unitree_hg.msg import (
    LowState,
    MotorState,
    IMUState,
    LowCmd,
    MotorCmd,
)

NUM_MOTOR = 29
HZ = 1000
crc = CRC()

dt = (1 / HZ) - 8e-6

class MessageRelayNode(Node):
    def __init__(self):
        super().__init__('message_relay_node')
        
        # Create a subscriber to listen to an input topic (such as 'input_topic')
        self.subscription = self.create_subscription(
            LowCmd,  # Replace with your message type
            'lowcmd_buffer',  # Replace with your input topic name
            self.listener_callback,
            1
        )
        
        # Create a publisher to publish messages to the 'lowcmd' topic
        self.publisher = self.create_publisher(LowCmd, 'lowcmd', 1)
        
        # Initialize a variable to store received messages
        self.last_msg = None
        self.last_last_msg = None
        self.counter = 0

        # init motor command
        self.new_msg = LowCmd()
        self.new_msg.mode_pr = 0
        self.new_msg.mode_machine = 5
        self.motor_cmd = []
        for id in range(NUM_MOTOR):
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=1, reserve=0)
            self.motor_cmd.append(cmd)
        for id in range(NUM_MOTOR, 35):
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=0, reserve=0)
            self.motor_cmd.append(cmd)
        self.new_msg.motor_cmd = self.motor_cmd.copy()

    def listener_callback(self, msg):
        # Save the message whenever one is received
        if self.last_msg is None:
            self.counter = 0
            self.last_msg = LowCmd()
            self.last_last_msg = LowCmd()
            self.last_msg.motor_cmd = msg.motor_cmd.copy()
            self.last_last_msg.motor_cmd = msg.motor_cmd.copy()
        else:
            self.counter = 0
            self.last_last_msg.motor_cmd = self.last_msg.motor_cmd.copy() # type: ignore
            self.last_msg.motor_cmd = msg.motor_cmd.copy()

    
    def set_motor_position(
        self,
    ):
        for i in range(NUM_MOTOR):
            # self.motor_cmd[i].q = self.last_msg.motor_cmd[i].q * (count/15) + self.last_last_msg.motor_cmd[i].q * (1-count/15)
            self.motor_cmd[i].q = self.last_msg.motor_cmd[i].q # type: ignore
            self.motor_cmd[i].kp = self.last_msg.motor_cmd[i].kp # type: ignore
            self.motor_cmd[i].kd = self.last_msg.motor_cmd[i].kd # type: ignore
        self.new_msg.motor_cmd = self.motor_cmd.copy()
        # self.cmd_msg.crc = get_crc(self.cmd_msg)
        self.new_msg.crc = crc.Crc(self.new_msg) # type: ignore
        self.counter += 1

    def relay_message(self):
        # If a message has been received, publish a message to the lowcmd topic each time this function is called
        if self.last_msg is not None:
            self.set_motor_position()
            self.publisher.publish(self.new_msg)
            # breakpoint()

def main(args=None):
    rclpy.init(args=args)
    
    # Create node instance
    node = MessageRelayNode()
    
    print('Initialized')
    try:
        loop_start_time = time.monotonic()
        while rclpy.ok():
            # Process subscription callback functions
            rclpy.spin_once(node, timeout_sec=0)
            
            # Forward message to lowcmd topic
            node.relay_message()
            
            while dt-time.monotonic()+loop_start_time>0:  #0.012473  0.019963 # Create 1000Hz publishing frequency
                pass
            loop_start_time = time.monotonic()
    
    except KeyboardInterrupt:
        pass
    
    # Close node
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()