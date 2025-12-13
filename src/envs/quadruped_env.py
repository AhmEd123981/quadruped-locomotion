"""
Quadruped Environment - FIXED
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class QuadrupedEnv(gym.Env):
    """Quadruped environment"""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render=False, target_velocity=0.5, reward_type='balanced', 
                 time_step=0.01, max_episode_steps=1000):
        super().__init__()
        
        self.render_mode = render
        self.target_velocity = target_velocity
        self.reward_type = reward_type
        self.time_step = time_step
        self.max_episode_steps = max_episode_steps
        
        # Connect to PyBullet
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("quadruped.urdf", [0, 0, 0.3])
        
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
        
        self.num_motors = len(self.joint_indices)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_motors,), dtype=np.float32)
        
        self.steps = 0
        self.episode_reward = 0.0
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("quadruped.urdf", [0, 0, 0.3])
        
        for joint_idx in self.joint_indices:
            p.resetJointState(self.robot_id, joint_idx, 0.0)
        
        self.steps = 0
        self.episode_reward = 0.0
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(action):
                target_pos = action[i] * np.pi / 4
                p.setJointMotorControl2(self.robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=target_pos, force=50.0)
        
        p.stepSimulation()
        self.steps += 1
        
        obs = self._get_observation()
        reward = self._compute_reward(obs, action)
        self.episode_reward += reward
        
        done = self._is_done()
        
        info = {
            'episode': {'r': self.episode_reward, 'l': self.steps},
            'speed': self._get_forward_velocity(),
            'gait_stability': self._compute_gait_stability(),
            'success': not self._is_fallen()
        }
        
        return obs, reward, done, False, info
    
    def _get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        joint_states = []
        for joint_idx in self.joint_indices:
            state = p.getJointState(self.robot_id, joint_idx)
            joint_states.append(state[0])
            joint_states.append(state[1])
        
        while len(joint_states) < 24:
            joint_states.append(0.0)
        
        obs = np.concatenate([pos, euler, lin_vel, ang_vel, joint_states[:24]])
        obs = obs[:37]
        while len(obs) < 37:
            obs = np.append(obs, 0.0)
        
        return obs.astype(np.float32)
    
    def _get_forward_velocity(self):
        _, lin_vel = p.getBaseVelocity(self.robot_id)
        return float(lin_vel[0])
    
    def _compute_gait_stability(self):
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, _ = euler
        stability = 1.0 - (abs(roll) + abs(pitch)) / np.pi
        return max(0.0, float(stability))
    
    def _is_fallen(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        if pos[2] < 0.15:
            return True
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, _ = euler
        if abs(roll) > np.pi / 3 or abs(pitch) > np.pi / 3:
            return True
        return False
    
    def _is_done(self):
        if self._is_fallen():
            return True
        if self.steps >= self.max_episode_steps:
            return True
        return False
    
    def _compute_reward(self, obs, action):
        speed = self._get_forward_velocity()
        stability = self._compute_gait_stability()
        
        if self.reward_type == 'balanced':
            speed_error = abs(speed - self.target_velocity)
            speed_reward = np.exp(-2.0 * speed_error)
            stability_reward = stability
            action_penalty = 0.01 * np.sum(np.square(action))
            total = 0.6 * speed_reward + 0.4 * stability_reward - action_penalty
        else:
            speed_error = abs(speed - self.target_velocity)
            total = np.exp(-2.0 * speed_error)
        
        return float(total)
    
    def close(self):
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
    
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def render(self, mode='human'):
        pass
