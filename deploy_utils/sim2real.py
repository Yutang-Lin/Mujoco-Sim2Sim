# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Yutang-Lin.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import mujoco
import mujoco.viewer
import numpy as np  
import torch
import time
import sys
import os
import time
from copy import deepcopy
from .math_utils import (
    euler_xyz_from_quat,
    quat_apply_inverse,
)
import rclpy
from rclpy.node import Node
from unitree_hg.msg import (
    LowState,
    MotorState,
    IMUState,
    LowCmd,
    MotorCmd,
)
from .crc import CRC

class UnitreeEnv(Node):
    def __init__(self, control_freq: int = 100, 
                 joint_order: list[str] | None = None,
                 action_joint_names: list[str] | None = None,
                 release_time_delta: float = 0.0,
                 init_rclpy: bool = True,
                 spin_timeout: float = 0.001,
                 **kwargs):
        """
        Initialize MuJoCo environment
        
        Args:
            control_freq: Control frequency in Hz (must be <= simulation_freq)
            joint_order: List of joint names specifying the order of joints for control and observation.
            action_joint_names: List of joint names that are actuated (subset of joint_order).
            release_time_delta: Delta time to step_complete return True before control dt reach
            init_rclpy: Whether to initialize rclpy
            spin_timeout: Timeout for rclpy.spin_once
        """
        self.control_freq = control_freq
        self.control_dt = 1.0 / self.control_freq
        self.release_time_delta = release_time_delta
        self.spin_timeout = spin_timeout
        
        # State variables
        self.step_count = 0

        assert joint_order is not None, "joint_order must be provided"
        
        # PD gains (can be modified) - now per-joint arrays
        self.kp = np.full(len(joint_order), 0.0)  # Default position gain for all joints
        self.kd = np.full(len(joint_order), 0.0)   # Default velocity gain for all joints

        # Get joint names
        self.joint_names = joint_order
        # Get body names
        self.body_names = None

        # Set num joints
        self.num_joints = len(joint_order)

        # Initiate ROS2 node
        if init_rclpy:
            rclpy.init()
        super().__init__('unitree_env')

        # Create a subscriber to listen to an input topic (such as 'input_topic')
        self.lowstate_sub = self.create_subscription(
            LowState,  # Replace with your message type
            'lowstate',  # Replace with your input topic name
            self._lowstate_callback,
            1
        )

        # Create a publisher to publish to an output topic (such as 'output_topic')
        self.lowcmd_pub = self.create_publisher(
            LowCmd,  # Replace with your message type
            'lowcmd_buffer',  # Replace with your output topic name
            1
        )

        # Create command
        self.lowcmd = LowCmd()
        self.lowcmd_initialized = False

        # Create motor commands
        self.motor_cmd = []
        for id in range(self.num_joints):
            self.motor_cmd.append(MotorCmd(mode=1, reserve=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0))
        for id in range(self.num_joints, 35):
            self.motor_cmd.append(MotorCmd(mode=0, reserve=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0))
        self.lowcmd.motor_cmd = self.motor_cmd.copy()

        # Initialize state buffers
        self.joint_pos = torch.zeros(self.num_joints)
        self.joint_vel = torch.zeros(self.num_joints)
        self.root_rpy = torch.zeros(3)
        self.root_quat = torch.zeros(4)
        self.root_ang_vel = torch.zeros(3)

        # Initialize target positions
        self.target_positions = torch.zeros(self.num_joints)
        self.last_publish_time = time.monotonic()

        # Initialize CRC
        self.crc = CRC()

        # Get joint order
        self.joint_order_names = joint_order
        self.joint_order = list(range(len(joint_order)))

        self.action_joint_names = action_joint_names
        self.action_joints = []
        if action_joint_names is None or len(action_joint_names) == 0:
            self.action_joints = deepcopy(self.joint_order)
        else:
            for name in action_joint_names:
                self.action_joints.append(self.joint_order_names.index(name))
        
        print(f"UnitreeEnv initialized:")
        print(f"  Joints: {len(self.joint_order_names)}")
        print(f"  Control frequency: {control_freq} Hz")
        print(f"  Default PD gains: kp={self.kp[0]}, kd={self.kd[0]} (for all joints)")

    def _lowstate_callback(self, msg: LowState):
        """Callback for lowstate topic"""
        self.lowstate = msg
        if not self.lowcmd_initialized:
            self.lowcmd_initialized = True
            self.lowcmd.mode_pr = msg.mode_pr
            self.lowcmd.mode_machine = msg.mode_machine
            motor_cmds = [x for x in msg.motor_state if x.mode == 1]
            assert len(motor_cmds) == self.num_joints, f"Expected {self.num_joints} motor commands, got {len(motor_cmds)}"

        self.joint_pos[:] = torch.tensor([x.q for x in motor_cmds]).float()
        self.joint_vel[:] = torch.tensor([x.dq for x in motor_cmds]).float()
        self.root_rpy[:] = torch.tensor([msg.imu_state.rpy[0], msg.imu_state.rpy[1], msg.imu_state.rpy[2]]).float()    
        self.root_quat[:] = torch.tensor([msg.imu_state.quaternion[3], msg.imu_state.quaternion[0], msg.imu_state.quaternion[1], msg.imu_state.quaternion[2]]).float()
        self.root_ang_vel[:] = torch.tensor([msg.imu_state.gyroscope[0], msg.imu_state.gyroscope[1], msg.imu_state.gyroscope[2]]).float()

    def reset(self, fix_root=False):
        """
        Reset the robot to initial state
        
        Args:
            fix_root: If True, fix the root joint to make the robot static/floating
        """
        # no-ops in sim2real
        pass

    def refresh_data(self):
        """Refresh data"""
        # Refresh data by spinning once
        rclpy.spin_once(self, timeout_sec=self.spin_timeout)
    
    def get_joint_data(self):
        """
        Get current joint data
        
        Returns:
            dict: Dictionary containing joint positions, velocities, and accelerations
        """

        return {
            'joint_pos': self.joint_pos.clone(),  # Joint positions
            'joint_vel': self.joint_vel.clone(),  # Joint velocities
        }
    
    def get_root_data(self):
        """
        Get current root data
        """
        return {
            'root_rpy': self.root_rpy.clone(),  # Root euler (x, y, z)
            'root_quat': self.root_quat.clone(),  # Root orientation (quaternion)
            'root_ang_vel': self.root_ang_vel.clone(),  # Root angular velocity
        }
    
    def get_body_data(self):
        """
        Get current body data
        
        Returns:
            dict: Dictionary containing body positions, orientations, and velocities
        """
        # Get body positions and orientations
        raise NotImplementedError("Body data not implemented for sim2real, consider LiDAR plugin.")
    
    def set_pd_gains(self, kp=None, kd=None):
        """
        Set PD control gains
        
        Args:
            kp: Position gain(s). Can be:
                - scalar: applied to all joints
                - array: per-joint gains (length must match num_joints)
                - None: keeps current value
            kd: Velocity gain(s). Can be:
                - scalar: applied to all joints  
                - array: per-joint gains (length must match num_joints)
                - None: keeps current value
        """
        if kp is not None:
            if np.isscalar(kp):
                self.kp = np.full(self.num_joints, kp)
            else:
                if isinstance(kp, torch.Tensor):
                    kp = kp.cpu().numpy()
                elif isinstance(kp, list):
                    kp = np.array(kp)
                assert isinstance(kp, np.ndarray)
                if len(kp) == self.num_joints:
                    self.kp = kp.copy()
                elif len(kp) == len(self.joint_order):
                    self.kp[self.joint_order] = kp.copy()
                    assert isinstance(self.joint_order_names, list)
                    remain_joints = set(self.joint_names) - set(self.joint_order_names)
                    print(f"Remaining joints for kpkd: {remain_joints}")
                elif len(kp) == len(self.action_joints):
                    self.kp[self.action_joints] = kp.copy()
                    assert isinstance(self.action_joint_names, list)
                    remain_joints = set(self.joint_names) - set(self.action_joint_names)
                    print(f"Remaining joints for kpkd: {remain_joints}")
                else:
                    raise ValueError(f"Expected kp array of length {self.num_joints}, got {len(kp)}")
        
        if kd is not None:
            if np.isscalar(kd):
                self.kd = np.full(self.num_joints, kd)
            else:
                if isinstance(kd, torch.Tensor):
                    kd = kd.cpu().numpy()
                elif isinstance(kd, list):
                    kd = np.array(kd)
                assert isinstance(kd, np.ndarray)
                if len(kd) == self.num_joints:
                    self.kd = kd.copy()
                elif len(kd) == len(self.joint_order):
                    self.kd[self.joint_order] = kd.copy()
                elif len(kd) == len(self.action_joints):
                    self.kd[self.action_joints] = kd.copy()
                else:
                    raise ValueError(f"Expected kd array of length {self.num_joints}, got {len(kd)}")
                
        for i in range(self.num_joints):
            self.motor_cmd[i].kp = float(self.kp[i])
            self.motor_cmd[i].kd = float(self.kd[i])
        
        print(f"Set PD gains:")
        print(f"  kp: {self.kp}")
        print(f"  kd: {self.kd}")
    
    def get_pd_gains(self, return_full=False):
        """
        Get current PD gains
        
        Returns:
            tuple: (kp_array, kd_array) current PD gains for all joints
        """
        if return_full:
            return torch.from_numpy(self.kp.copy()).float(), torch.from_numpy(self.kd.copy()).float()
        else:
            return torch.from_numpy(self.kp[self.joint_order].copy()).float(), torch.from_numpy(self.kd[self.joint_order].copy()).float()
    
    def apply_pd_control(self):
        """Apply PD control using current target positions"""
        # For position actuators, we just set the target positions
        # MuJoCo will automatically apply PD control
        for i in range(self.num_joints):
            self.motor_cmd[i].q = self.target_positions[i].item()
        self.lowcmd.motor_cmd = self.motor_cmd.copy()
        self.lowcmd.crc = self.crc.Crc(self.lowcmd) # type: ignore
        self.lowcmd_pub.publish(self.lowcmd)

    def step_complete(self):
        """Check if the simulation step is complete"""
        step_complete = time.monotonic() - self.last_publish_time > self.control_dt - self.release_time_delta
        if step_complete:
            rclpy.spin_once(self, timeout_sec=self.spin_timeout)
        return step_complete
    
    def step(self, actions=None):
        """
        Step the simulation forward by running decimation number of simulation steps
        
        Args:
            actions: Optional array of target positions for joints (excluding root)
                    If provided, updates the target positions before applying control
        Returns:
            bool: True if simulation is still running, False if it should stop
        """
        # Update target positions if provided
        if actions is not None:
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)
            elif isinstance(actions, list):
                actions = torch.tensor(actions)
            assert isinstance(actions, torch.Tensor)
            if len(actions) != len(self.action_joints):
                raise ValueError(f"Expected actions array of length {len(self.action_joints)}, got {len(actions)}")
            self.target_positions[self.action_joints] = actions.clone().float()
        
        # Apply PD control
        self.apply_pd_control()
        self.last_publish_time = time.monotonic()

        return True
    
    def run_simulation(self, max_steps=None):
        """
        Run the simulation loop
        
        Args:
            max_steps: Maximum number of steps to run (None for infinite)
        """
        # Error in sim2real
        raise NotImplementedError("Run simulation not implemented for sim2real")
    
    def close(self):
        """Close the environment and cleanup resources"""
        # Error in sim2real
        self.destroy_node()

def main():
    """Main function to run the sim2real"""
    rclpy.init()

    joint_names = [
        "left_hip_yaw_joint",
        "left_hip_pitch_joint", 
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint", 
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint", 
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
        "right_wrist_pitch_joint", 
        "right_wrist_yaw_joint",
    ]

    # Create environment
    env = UnitreeEnv(
        control_freq=50,
        joint_order=joint_names,
    )
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()

