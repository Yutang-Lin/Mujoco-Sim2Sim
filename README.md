# Mujoco-Sim2Sim
## Introduction
Simple implementation of Gym stype Mujoco env for sim2sim. Along with a ROS2 env for Unitree robots' real deployment with same interface as Mujoco env.

## Usage
Format your server as following to get seamlessly transfer from sim2sim to sim2real, and wrapped ROS2 communication.
```python
from deploy_utils.sim2sim import MujocoEnv
# from deploy_utils.sim2real import UnitreeEnv # for real deployment
"""initialize env, setup control hz etc."""
env = MujocoEnv(...) # UnitreeEnv(...) # switch freely
"""control loop for your policy"""
while not end_condition:
    while not env.step_complete():
        time.sleep(0.001)
    env.refresh_data()

    joint_data = env.get_joint_data() # dict of joint_pos, joint_vel
    root_data = env.get_root_data() # dict of root_rpy, root_quat, root_ang_vel
    # body_data = env.get_body_data() # dict of body_pos ..., only in sim2sim

    obs = compute_observation(joint_data, root_data)
    actions = policy(actions)
    actions = (actions - default_actions) * action_scale # etc.

    env.step(actions)

    """for sim2sim"""
    if terminate_condition:
        env.reset(fix_root=True) # reset as you needed in sim2sim scenarios
```