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

class MujocoEnv:
    def __init__(self, control_freq: int = 100, 
                 joint_order: list[str] | None = None,
                 action_joint_names: list[str] | None = None,
                 model_path: str = '', 
                 simulation_freq: int = 1000,
                 joint_armature: float = 0.01, 
                 joint_damping: float = 0.1, 
                 enable_viewer: bool = True,
                 enable_ros_control: bool = False,
                 **kwargs):
        """
        Initialize MuJoCo environment
        
        Args:
            control_freq: Control frequency in Hz (must be <= simulation_freq)
            joint_order: List of joint names specifying the order of joints for control and observation.
            action_joint_names: List of joint names that are actuated (subset of joint_order).
            model_path: Path to the MuJoCo XML model file
            simulation_freq: Simulation frequency in Hz
            joint_armature: Joint armature (motor inertia) value
            joint_damping: Joint damping value
            enable_viewer: Whether to enable the MuJoCo viewer
        """
        self.model_path = model_path
        self.control_freq = control_freq
        self.simulation_freq = simulation_freq
        self.joint_armature = joint_armature
        self.joint_damping = joint_damping
        self.enable_viewer = enable_viewer
        self.enable_ros_control = enable_ros_control

        # Validate control frequency
        if control_freq > simulation_freq:
            raise ValueError(f"Control frequency ({control_freq} Hz) cannot be higher than simulation frequency ({simulation_freq} Hz)")
        
        # Calculate decimation (how many simulation steps per control step)
        self.decimation = int(simulation_freq / control_freq)
        
        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(model_path) # type: ignore
        self.data = mujoco.MjData(self.model) # type: ignore
        
        # Setup simulation
        self._setup_joint_armature()
        self._setup_joint_damping(joint_damping)
        self._setup_simulation_frequency()
        
        # Initialize viewer if enabled
        self.viewer = None
        if self.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # State variables
        self.step_count = 0
        self.root_fixed = False
        
        # Initialize target positions (excluding root)
        self.num_joints = self.model.nu  # Total actuators
        self.target_positions = np.zeros(self.num_joints)
        
        # PD gains (can be modified) - now per-joint arrays
        self.kp = np.full(self.num_joints, 0.0)  # Default position gain for all joints
        self.kd = np.full(self.num_joints, 0.0)   # Default velocity gain for all joints

        # Get joint names
        self._joint_names = [self.model.joint(i).name for i in range(self.model.njnt)][1:]
        # Get actuator names
        self.joint_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        # Get body names
        self.body_names = [self.model.body(i).name for i in range(self.model.nbody)]

        # Get joint order
        self.joint_order_names = joint_order
        self.joint_order = []
        if joint_order is None or len(joint_order) == 0:
            self.joint_order = list(range(self.num_joints))
        else:
            for name in joint_order:
                self.joint_order.append(self.joint_names.index(name))

        self._joint_order = []
        if joint_order is None or len(joint_order) == 0:
            self._joint_order = [self.joint_names.index(name) for name in self._joint_names]
        else:
            for name in joint_order:
                self._joint_order.append(self._joint_names.index(name))

        self.action_joint_names = action_joint_names
        self.action_joints = []
        if action_joint_names is None or len(action_joint_names) == 0:
            self.action_joints = deepcopy(self.joint_order)
        else:
            for name in action_joint_names:
                self.action_joints.append(self.joint_names.index(name))
        
        print(f"MuJoCo Environment initialized:")
        print(f"  Model: {model_path}")
        print(f"  Joints: {self.num_joints} (excluding root)")
        print(f"  Actuators: {len(self.data.ctrl)}")
        print(f"  Simulation frequency: {simulation_freq} Hz")
        print(f"  Control frequency: {control_freq} Hz")
        print(f"  Decimation: {self.decimation} simulation steps per control step")
        print(f"  Armature: {joint_armature}")
        print(f"  Default PD gains: kp={self.kp[0]}, kd={self.kd[0]} (for all joints)")
    
    def _setup_joint_armature(self):
        """Set joint armature (motor inertia) for all joints"""
        self.model.dof_armature[:] = self.joint_armature
        print(f"Set joint armature to {self.joint_armature} for all joints")
    
    def _setup_joint_damping(self, damping=0.1):
        """Set joint damping for all joints"""
        self.model.dof_damping[:] = damping
        print(f"Set joint damping to {damping} for all joints")
    
    def _setup_simulation_frequency(self):
        """Set simulation frequency"""
        dt = 1.0 / self.simulation_freq
        self.model.opt.timestep = dt
        print(f"Set simulation frequency to {self.simulation_freq} Hz (timestep: {dt:.6f} s)")

    def _update_model_actuator_gains(self):
        """Update MuJoCo model actuator gains if using position actuators"""
        # Check if any actuators are position actuators
        for i in range(self.model.nu):
            # Update the model's actuator gains
            self.model.actuator_gainprm[i] = self.kp[i]  # kp
            self.model.actuator_biasprm[i] = self.kd[i]  # kd
        print(f"Updated model actuator gains: kp={self.kp}, kd={self.kd}")

    def reset(self, fix_root=False):
        """
        Reset the robot to initial state
        
        Args:
            fix_root: If True, fix the root joint to make the robot static/floating
        """
        # Reset all joint positions and velocities
        self.data.qpos[:] = self.model.qpos0[:]
        self.data.qvel[:] = 0.0
        
        # If fix_root is True, fix the root joint to make the robot static
        if fix_root:
            # Set root position to initial position
            self.data.qpos[0:3] = self.model.qpos0[0:3]  # x, y, z position
            self.data.qpos[3:7] = self.model.qpos0[3:7]  # quaternion orientation
            
            # Set root velocity to zero to stop any movement
            self.data.qvel[0:6] = 0.0  # linear and angular velocity of root
        
        # Reset time
        self.data.time = 0.0
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data) # type: ignore
        
        # Update internal state
        self.root_fixed = fix_root
        self.step_count = 0
        
        print(f"Robot reset! Root {'fixed (floating)' if fix_root else 'free'}")

    def refresh_data(self):
        """Refresh data"""
        # refresh data is no-ops for sim2sim
        pass
    
    def get_joint_data(self):
        """
        Get current joint data
        
        Returns:
            dict: Dictionary containing joint positions, velocities.
        """
        return {
            'joint_pos': torch.from_numpy(self.data.qpos[7:].copy()[self._joint_order]).float(),  # Joint positions (excluding root)
            'joint_vel': torch.from_numpy(self.data.qvel[6:].copy()[self._joint_order]).float(),  # Joint velocities (excluding root)
        }
    
    def get_root_data(self):
        """
        Get current root data, including root orientation, and relative angular velocity.
        """
        root_quat = torch.from_numpy(self.data.qpos[3:7].copy()).float()
        root_ang_vel = torch.from_numpy(self.data.qvel[3:6].copy()).float()
        root_ang_vel = quat_apply_inverse(root_quat, root_ang_vel)
        root_rpy = torch.stack(euler_xyz_from_quat(root_quat.view(1, 4)), dim=-1).view(-1)

        return {
            'root_rpy': root_rpy,  # Root euler (x, y, z)
            'root_quat': root_quat,  # Root orientation (quaternion)
            'root_ang_vel': root_ang_vel,  # Root angular velocity
        }
    
    def get_body_data(self):
        """
        Get current body data
        
        Returns:
            dict: Dictionary containing body positions, orientations, and velocities
        """
        # Get body positions and orientations
        body_positions = []
        body_orientations = []
        body_velocities = []
        body_angular_velocities = []
        
        for i in range(self.model.nbody):
            # Get body position and orientation
            pos = self.data.xpos[i].copy()
            quat = self.data.xquat[i].copy()
            
            # Get body velocity (linear and angular)
            vel = self.data.cvel[i].copy()  # [linear_vel, angular_vel]
            
            body_positions.append(pos)
            body_orientations.append(quat)
            body_velocities.append(vel[:3])  # Linear velocity
            body_angular_velocities.append(vel[3:])  # Angular velocity
        
        return {
            'body_pos': torch.from_numpy(np.array(body_positions)).float(),
            'body_quat': torch.from_numpy(np.array(body_orientations)).float(),
            'body_lin_vel': torch.from_numpy(np.array(body_velocities)).float(),
            'body_ang_vel': torch.from_numpy(np.array(body_angular_velocities)).float(),
        }

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
        
        # Update MuJoCo model actuator gains if using position actuators
        self._update_model_actuator_gains()
        
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
        """Apply PD control using current target positions and PD gains"""
        # Get current joint positions and velocities (excluding root)
        current_positions = self.data.qpos[7:].copy()[self._joint_order]  # Joint positions (excluding root)
        current_velocities = self.data.qvel[6:].copy()[self._joint_order]  # Joint velocities (excluding root)
        
        # Get target positions for the joints we're controlling
        target_positions = self.target_positions[self.joint_order]
        
        # Compute PD control torques: tau = kp * (target_pos - current_pos) + kd * (0 - current_vel)
        position_errors = target_positions - current_positions
        velocity_errors = 0.0 - current_velocities  # Target velocity is 0
        
        # Get PD gains for the joints we're controlling
        kp_control = self.kp[self.joint_order]
        kd_control = self.kd[self.joint_order]
        
        # Compute control torques
        control_torques = kp_control * position_errors + kd_control * velocity_errors
        
        # Apply torques to the actuators
        self.data.ctrl[self.joint_order] = control_torques

    def step_complete(self):
        """Check if the simulation step is complete"""
        return True
    
    def step(self, actions=None):
        """
        Step the simulation forward by running decimation number of simulation steps
        
        Args:
            actions: Optional array of target positions for joints (excluding root)
                    If provided, updates the target positions before applying control
        Returns:
            bool: True if simulation is still running, False if it should stop
        """
        control_period = self.decimation / self.simulation_freq  # seconds per control step
        start_time = time.time()
        # Update target positions if provided
        if actions is not None:
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            elif isinstance(actions, list):
                actions = np.array(actions)
            assert isinstance(actions, np.ndarray)
            if len(actions) != len(self.action_joints):
                raise ValueError(f"Expected actions array of length {len(self.action_joints)}, got {len(actions)}")
            self.target_positions[self.action_joints] = actions.copy()
        
        # Run decimation number of simulation steps
        for _ in range(self.decimation):
            # If root is fixed, keep it at the initial position
            if self.root_fixed:
                self.data.qpos[0:7] = self.model.qpos0[0:7]  # Keep root position and orientation fixed
                self.data.qvel[0:6] = 0.0  # Keep root velocity at zero
            
            # Apply PD control
            self.apply_pd_control()
            
            # Check for simulation instability
            if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)):
                print("Warning: Simulation becoming unstable, resetting...")
                self.reset(fix_root=self.root_fixed)
                return True
            
            # Step simulation
            mujoco.mj_step(self.model, self.data) # type: ignore
            
            # Sync viewer if enabled (only on the last step to avoid excessive syncing)
            if self.viewer is not None and _ == self.decimation - 1:
                self.viewer.sync()
            
            self.step_count += 1
        # Real-time synchronization
        elapsed = time.time() - start_time
        if elapsed < control_period:
            time.sleep(control_period - elapsed)
        return True
    
    def _set_random_targets(self, range_min=-0.2, range_max=0.2):
        """
        Set random target positions
        
        Args:
            range_min: Minimum value for random targets
            range_max: Maximum value for random targets
        """
        self.target_positions = np.random.uniform(range_min, range_max, self.num_joints)
        print(f"Set random target positions (range: {range_min} to {range_max})")
    
    def _set_zero_targets(self):
        """Set all target positions to zero"""
        self.target_positions = np.zeros(self.num_joints)
        print("Set zero target positions")
    
    def run_simulation(self, max_steps=None):
        """
        Run the simulation loop
        
        Args:
            max_steps: Maximum number of steps to run (None for infinite)
        """
        print("Simulation starting...")
        print("Commands (type in terminal):")
        print("  'r' - Reset robot with free root")
        print("  'rf' - Reset robot with fixed root (floating)") 
        print("  'q' - Quit simulation")
        print("  'h' - Show this help")
        print("  'p' - Set random target positions")
        print("  'z' - Set zero target positions")
        print("\nNote: You can type commands in the terminal while simulation is running.")
        
        try:
            while (self.viewer is None or self.viewer.is_running()) and (max_steps is None or self.step_count < max_steps):
                # Check for input every 100 steps (roughly every 0.1 seconds at 1kHz)
                if self.step_count % 100 == 0:
                    try:
                        # Use a non-blocking input check
                        import select
                        if select.select([sys.stdin], [], [], 0)[0]:
                            user_input = sys.stdin.readline().strip().lower()
                            if user_input == 'r':
                                self.reset(fix_root=False)
                            elif user_input == 'rf':
                                self.reset(fix_root=True)
                            elif user_input == 'q':
                                print("Quitting simulation...")
                                break
                            elif user_input == 'h':
                                print("Commands:")
                                print("  'r' - Reset robot with free root")
                                print("  'rf' - Reset robot with fixed root (floating)") 
                                print("  'q' - Quit simulation")
                                print("  'h' - Show this help")
                                print("  'p' - Set random target positions")
                                print("  'z' - Set zero target positions")
                            elif user_input == 'p':
                                self._set_random_targets()
                            elif user_input == 'z':
                                self._set_zero_targets()
                    except:
                        pass  # Ignore input errors
                
                # Step simulation
                if not self.step():
                    break
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("Simulation ended.")
    
    def close(self):
        """Close the environment and cleanup resources"""
        if self.viewer is not None:
            self.viewer.close()
        print("Environment closed.")


def main():
    """Main function to run the simulation"""
    abs_path = os.path.abspath(__file__)
    abs_dir = os.path.dirname(abs_path)

    # Create environment
    env = MujocoEnv(
        model_path=os.path.join(abs_dir, 'assets/h1_2.xml'),
        simulation_freq=1000,
        control_freq=50,
        joint_armature=0.1,
        joint_damping=1.0,
        enable_viewer=True
    )
    
    # Run simulation
    env.run_simulation()
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()

