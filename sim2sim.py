import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
import time

class MujocoEnv:
    def __init__(self, model_path: str, 
                 simulation_freq: int = 1000, 
                 control_freq: int = 100, 
                 joint_armature: float = 0.01, 
                 joint_damping: float = 0.1, 
                 enable_viewer: bool = True):
        """
        Initialize MuJoCo environment
        
        Args:
            model_path: Path to the MuJoCo XML model file
            simulation_freq: Simulation frequency in Hz
            control_freq: Control frequency in Hz (must be <= simulation_freq)
            joint_armature: Joint armature (motor inertia) value
            joint_damping: Joint damping value
            enable_viewer: Whether to enable the MuJoCo viewer
        """
        self.model_path = model_path
        self.simulation_freq = simulation_freq
        self.control_freq = control_freq
        self.joint_armature = joint_armature
        self.joint_damping = joint_damping
        self.enable_viewer = enable_viewer
        
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
        self.num_joints = self.model.nq - 7  # Total DOF minus root (7 DOF)
        self.target_positions = np.zeros(self.num_joints)
        
        # PD gains (can be modified) - now per-joint arrays
        self.kp = np.full(self.num_joints, 100.0)  # Default position gain for all joints
        self.kd = np.full(self.num_joints, 10.0)   # Default velocity gain for all joints

        # Get joint names
        # ignore floating base
        self.joint_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        
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
    
    def get_joint_data(self):
        """
        Get current joint data
        
        Returns:
            dict: Dictionary containing joint positions, velocities, and accelerations
        """
        return {
            'qpos': self.data.qpos.copy(),  # All joint positions (including root)
            'qvel': self.data.qvel.copy(),  # All joint velocities (including root)
            'qacc': self.data.qacc.copy(),  # All joint accelerations (including root)
            'joint_pos': self.data.qpos[7:].copy(),  # Joint positions (excluding root)
            'joint_vel': self.data.qvel[6:].copy(),  # Joint velocities (excluding root)
            'joint_acc': self.data.qacc[6:].copy(),  # Joint accelerations (excluding root)
            'root_pos': self.data.qpos[0:3].copy(),  # Root position (x, y, z)
            'root_quat': self.data.qpos[3:7].copy(),  # Root orientation (quaternion)
            'root_lin_vel': self.data.qvel[0:3].copy(),  # Root linear velocity
            'root_ang_vel': self.data.qvel[3:6].copy(),  # Root angular velocity
            'joint_names': self.joint_names
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
            'body_pos': np.array(body_positions),
            'body_quat': np.array(body_orientations),
            'body_lin_vel': np.array(body_velocities),
            'body_ang_vel': np.array(body_angular_velocities),
            'body_names': [self.model.body(i).name for i in range(self.model.nbody)]
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
                kp = np.array(kp)
                if len(kp) != self.num_joints:
                    raise ValueError(f"Expected kp array of length {self.num_joints}, got {len(kp)}")
                self.kp = kp.copy()
        
        if kd is not None:
            if np.isscalar(kd):
                self.kd = np.full(self.num_joints, kd)
            else:
                kd = np.array(kd)
                if len(kd) != self.num_joints:
                    raise ValueError(f"Expected kd array of length {self.num_joints}, got {len(kd)}")
                self.kd = kd.copy()
        
        print(f"Set PD gains:")
        print(f"  kp: {self.kp}")
        print(f"  kd: {self.kd}")
    
    def get_pd_gains(self):
        """
        Get current PD gains
        
        Returns:
            tuple: (kp_array, kd_array) current PD gains for all joints
        """
        return self.kp.copy(), self.kd.copy()
    
    def set_target_positions(self, actions):
        """
        Set target positions for all joints (excluding root)
        
        Args:
            actions: Array of target positions for joints
        """
        if len(actions) != self.num_joints:
            raise ValueError(f"Expected actions array of length {self.num_joints}, got {len(actions)}")
        
        self.target_positions = np.array(actions).copy()
    
    def set_random_targets(self, range_min=-0.2, range_max=0.2):
        """
        Set random target positions
        
        Args:
            range_min: Minimum value for random targets
            range_max: Maximum value for random targets
        """
        self.target_positions = np.random.uniform(range_min, range_max, self.num_joints)
        print(f"Set random target positions (range: {range_min} to {range_max})")
    
    def set_zero_targets(self):
        """Set all target positions to zero"""
        self.target_positions = np.zeros(self.num_joints)
        print("Set zero target positions")
    
    def apply_pd_control(self):
        """Apply PD control using current target positions"""
        # For position actuators, we just set the target positions
        # MuJoCo will automatically apply PD control
        self.data.ctrl[:] = self.target_positions
    
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
            actions = np.array(actions)
            if len(actions) != self.num_joints:
                raise ValueError(f"Expected actions array of length {self.num_joints}, got {len(actions)}")
            self.target_positions = actions.copy()
        
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
                                self.set_random_targets()
                            elif user_input == 'z':
                                self.set_zero_targets()
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

