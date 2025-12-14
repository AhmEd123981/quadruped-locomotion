

import numpy as np


def compute_speed_reward(obs, target_velocity):
    
    # Extract forward velocity from observation
    forward_vel = obs[6]  # Assuming velocity is at index 6
    
    # Reward is based on how close to target velocity
    speed_error = abs(forward_vel - target_velocity)
    speed_reward = np.exp(-speed_error)
    
    return speed_reward


def compute_balanced_reward(
    obs,
    action,
    target_velocity,
    speed_weight=0.6,
    gait_weight=0.4,
    reference_gait=None
):
   
    # 1. Speed component (task performance)
    forward_vel = obs[6]
    speed_error = abs(forward_vel - target_velocity)
    speed_reward = np.exp(-2.0 * speed_error)
    
    # 2. Gait stability component (preserving natural locomotion)
    gait_stability_reward = compute_gait_stability_reward(obs)
    
    # 3. Energy efficiency (penalize excessive actions)
    action_magnitude = np.sum(np.square(action))
    energy_penalty = 0.01 * action_magnitude
    
    # 4. Smoothness (penalize jerky movements)
    # Could compare with previous action if available
    smoothness_reward = 0.0  # Placeholder
    
    # 5. Reference gait preservation (if available from BC)
    gait_preservation_reward = 0.0
    if reference_gait is not None:
        gait_deviation = np.linalg.norm(action - reference_gait)
        gait_preservation_reward = np.exp(-gait_deviation)
    
    # Combine components with weights
    total_reward = (
        speed_weight * speed_reward +
        gait_weight * gait_stability_reward +
        0.1 * gait_preservation_reward -
        energy_penalty
    )
    
    return total_reward


def compute_gait_stability_reward(obs):
    
    # Extract base orientation (Euler angles)
    roll = obs[3]
    pitch = obs[4]
    
    # Extract base height
    height = obs[2]
    
    # Orientation stability (penalize large roll/pitch)
    orientation_stability = np.exp(-5.0 * (roll**2 + pitch**2))
    
    # Height stability (encourage maintaining proper height)
    target_height = 0.3
    height_error = abs(height - target_height)
    height_reward = np.exp(-10.0 * height_error)
    
    # Combine stability components
    stability_reward = 0.7 * orientation_stability + 0.3 * height_reward
    
    return stability_reward


def compute_forward_progress_reward(obs, prev_pos):
    
    current_pos = obs[0]  # x position
    forward_progress = current_pos - prev_pos
    
    # Reward positive progress, penalize backward motion
    return max(0.0, forward_progress)


def compute_energy_efficiency_reward(action, prev_action=None):
    
    # Action magnitude penalty
    magnitude_penalty = 0.01 * np.sum(np.square(action))
    
    # Action change penalty (smoothness)
    smoothness_penalty = 0.0
    if prev_action is not None:
        action_diff = action - prev_action
        smoothness_penalty = 0.005 * np.sum(np.square(action_diff))
    
    return -(magnitude_penalty + smoothness_penalty)


def compute_contact_reward(robot_id, ground_id):
    
    # This would require PyBullet integration
    # Placeholder for now
    return 0.0


def compute_fall_penalty(obs):
   
    height = obs[2]
    roll = obs[3]
    pitch = obs[4]
    
    # Large penalty if robot is too low or tilted
    if height < 0.15 or abs(roll) > np.pi/3 or abs(pitch) > np.pi/3:
        return -10.0
    
    return 0.0


def compute_composite_reward(
    obs,
    action,
    target_velocity,
    prev_pos=None,
    prev_action=None,
    weights=None
):
    
    if weights is None:
        weights = {
            'speed': 0.5,
            'stability': 0.3,
            'energy': 0.1,
            'progress': 0.1
        }
    
    reward = 0.0
    
    # Speed component
    if 'speed' in weights:
        speed_rew = compute_speed_reward(obs, target_velocity)
        reward += weights['speed'] * speed_rew
    
    # Stability component
    if 'stability' in weights:
        stability_rew = compute_gait_stability_reward(obs)
        reward += weights['stability'] * stability_rew
    
    # Energy efficiency
    if 'energy' in weights:
        energy_rew = compute_energy_efficiency_reward(action, prev_action)
        reward += weights['energy'] * energy_rew
    
    # Forward progress
    if 'progress' in weights and prev_pos is not None:
        progress_rew = compute_forward_progress_reward(obs, prev_pos)
        reward += weights['progress'] * progress_rew
    
    # Fall penalty
    fall_penalty = compute_fall_penalty(obs)
    reward += fall_penalty
    
    return reward
