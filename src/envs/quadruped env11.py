"""
Quadruped locomotion environment using PyBullet.
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from src.envs.reward_functions import compute_balanced_reward, compute_speed_reward


class QuadrupedEnv(gym.Env):
    """
    Custom quadruped locomotion environment.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        render=False,
        robot_urdf='quadruped.urdf',
        time_step=0.01,
        max_episode_steps=1000,
        target_velocity=0.5,
        reward_type='balanced',
        speed_weight=0.6,
        gait_weight=0.4
    ):
        super(QuadrupedEnv, self).__init__()
        
        self.render_mode = 'human' if render else None
        self.robot_urdf = robot_urdf
        self.time_step = time_step
        self.max_episode_steps = max_episode_steps
        self.target_velocity = target_velocity
        self.reward_type = reward_type
        self.speed_weight = speed_weight
        self.gait_weight = gait_weight
        
        # Connect to PyBullet
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Load plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = None
        
        # Robot parameters (adjust based on actual robot)
        self.num_joints = 12  # 3 joints per leg × 4 legs
        
        # Define observation and action spaces
        obs_dim = 37  # Adjust based on actual observations
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.steps = 0
        self.episode_reward = 0
        
        # Previous state for computing velocity and stability
        self.prev_base_pos = None
        self.prev_base_orn = None
        self.prev_action = None
        self.reference_gait = None  # To be loaded from BC model
        
        # Random seed
        self._np_random = None
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            self._np_random = np.random.RandomState(seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Reload plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot at starting position with small random variations
        start_pos = [0, 0, 0.3]
        if self._np_random is not None:
            start_pos[0] += self._np_random.uniform(-0.05, 0.05)
            start_pos[1] += self._np_random.uniform(-0.05, 0.05)
        
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Try to load custom URDF, fallback to laikago if not found
        try:
            self.robot_id = p.loadURDF(
                self.robot_urdf,
                start_pos,
                start_orientation
            )
        except:
            # Fallback to laikago robot
            try:
                self.robot_id = p.loadURDF(
                    "laikago/laikago.urdf",
                    start_pos,
                    start_orientation
                )
            except:
                # Last resort: use a simple box as robot
                import pybullet_data
                self.robot_id = p.loadURDF(
                    "r2d2.urdf",
                    start_pos,
                    start_orientation
                )
        
        # Get actual number of joints
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Reset joint positions to neutral stance
        for joint_idx in range(self.num_joints):
            p.resetJointState(self.robot_id, joint_idx, 0.0)
        
        # Reset tracking variables
        self.steps = 0
        self.episode_reward = 0
        self.prev_base_pos = np.array(start_pos)
        self.prev_base_orn = np.array(start_orientation)
        self.prev_action = None
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        # Convert action to numpy array if needed
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        # Handle scalar or 1D array
        if action.ndim == 0:
            action = np.array([action])
        
        """Execute one step in the environment."""
        # Apply action (convert from [-1, 1] to actual joint ranges)
        action = np.clip(action, -1.0, 1.0)
        
        # Ensure action matches number of joints
        if len(action) != self.num_joints:
            # Pad or truncate action to match joints
            if len(action) < self.num_joints:
                action = np.pad(action, (0, self.num_joints - len(action)))
            else:
                action = action[:self.num_joints]
        
        for joint_idx in range(self.num_joints):
            target_pos = float(action[joint_idx]) * np.pi / 4  # Scale to reasonable range
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=20.0
            )
        
        # Step simulation
        p.stepSimulation()
        self.steps += 1
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(obs, action)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._is_fallen()
        truncated = self.steps >= self.max_episode_steps
        
        # Additional info
        info = {
            'episode': {
                'r': self.episode_reward,
                'l': self.steps
            } if (terminated or truncated) else None,
            'speed': self._get_forward_velocity(),
            'gait_stability': self._compute_gait_stability(),
            'success': not self._is_fallen()
        }
        
        # Store previous action
        self.prev_action = action
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation from the environment."""
        # Base state
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Convert orientation to Euler angles
        base_euler = p.getEulerFromQuaternion(base_orn)
        
        # Joint states
        joint_states = []
        for joint_idx in range(min(12, self.num_joints)):  # Use up to 12 joints
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_states.extend([joint_state[0], joint_state[1]])  # position, velocity
        
        # Pad if fewer than 12 joints
        while len(joint_states) < 24:
            joint_states.extend([0.0, 0.0])
        
        # Construct observation vector (37 dimensions)
        obs = np.concatenate([
            base_pos,           # 3
            base_euler,         # 3
            base_vel,           # 3
            base_ang_vel,       # 3
            joint_states[:24],  # 24 (12 joints × 2)
            [self.steps / self.max_episode_steps]  # 1 (normalized time)
        ])
        
        # Ensure exactly 37 dimensions
        if len(obs) < 37:
            obs = np.pad(obs, (0, 37 - len(obs)))
        elif len(obs) > 37:
            obs = obs[:37]
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, obs, action):
        """Compute reward based on current state and action."""
        if self.reward_type == 'balanced':
            return compute_balanced_reward(
                obs,
                action,
                self.target_velocity,
                self.speed_weight,
                self.gait_weight,
                self.reference_gait
            )
        elif self.reward_type == 'speed':
            return compute_speed_reward(obs, self.target_velocity)
        else:
            # Default: simple forward velocity reward
            forward_vel = self._get_forward_velocity()
            return forward_vel
    
    def _get_forward_velocity(self):
        """Get forward (x-axis) velocity."""
        base_vel, _ = p.getBaseVelocity(self.robot_id)
        return base_vel[0]
    
    def _compute_gait_stability(self):
        """Compute gait stability metric."""
        # Simple stability measure based on base orientation
        _, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_euler = p.getEulerFromQuaternion(base_orn)
        
        # Penalize roll and pitch deviations
        roll, pitch, _ = base_euler
        stability = 1.0 - (abs(roll) + abs(pitch)) / np.pi
        
        return max(0.0, stability)
    
    def _is_fallen(self):
        """Check if robot has fallen."""
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # Check if base is too low
        if base_pos[2] < 0.15:
            return True
        
        # Check if roll or pitch is too large
        base_euler = p.getEulerFromQuaternion(base_orn)
        roll, pitch, _ = base_euler
        
        if abs(roll) > np.pi / 3 or abs(pitch) > np.pi / 3:
            return True
        
        return False
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'rgb_array':
            # Get camera image
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=2.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
            
            (_, _, px, _, _) = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (480, 640, 4))
            rgb_array = rgb_array[:, :, :3]
            
            return rgb_array
        
        return None
    
    def close(self):
        """Clean up environment."""
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
    
    def seed(self, seed=None):
        """Set random seed (for backward compatibility)."""
        self._np_random = np.random.RandomState(seed)
        np.random.seed(seed)
        return [seed]
