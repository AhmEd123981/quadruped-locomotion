"""
Generate realistic quadruped locomotion demonstration data.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict


class QuadrupedGaitGenerator:
    """Generates realistic quadruped locomotion trajectories."""
    
    def __init__(self, num_joints: int = 12, obs_dim: int = 37):
        self.num_joints = num_joints
        self.obs_dim = obs_dim
        self.dt = 0.01
        self.gait_frequency = 2.0
        self.stride_length = 0.3
        self.swing_height = 0.15
        self.base_height = 0.3
        
    def generate_trotting_gait(self, duration: float, phase_offset: float = 0.0) -> Dict:
        """Generate trotting gait (diagonal legs move together)."""
        num_steps = int(duration / self.dt)
        time = np.arange(num_steps) * self.dt
        observations = []
        actions = []
        
        base_x = 0.0
        base_z = self.base_height
        orientation = np.array([0.0, 0.0, 0.0])
        vel_x = self.stride_length * self.gait_frequency
        vel_z = 0.0
        
        for t in time:
            phase = (t * self.gait_frequency + phase_offset) % 1.0
            swing_phase = phase % 1.0
            is_swing = (0.3 < swing_phase < 0.7)
            
            joint_angles = np.zeros(self.num_joints)
            for leg_idx in range(4):
                if leg_idx % 2 == 0:
                    leg_is_swing = is_swing
                else:
                    leg_is_swing = not is_swing
                
                if leg_is_swing:
                    local_phase = (swing_phase - 0.3) / 0.4
                    local_phase = np.clip(local_phase, 0, 1)
                    hip_angle = 0.3 * np.sin(np.pi * local_phase)
                    knee_angle = -0.5 * np.sin(2 * np.pi * local_phase)
                    ankle_angle = 0.2 * np.cos(np.pi * local_phase)
                else:
                    local_phase = (swing_phase - 0.8) / 0.4
                    local_phase = np.clip(local_phase, 0, 1)
                    hip_angle = -0.2 * np.sin(np.pi * local_phase)
                    knee_angle = 0.3 * np.cos(np.pi * local_phase)
                    ankle_angle = -0.1 * np.sin(np.pi * local_phase)
                
                joint_angles[leg_idx * 3 + 0] = hip_angle
                joint_angles[leg_idx * 3 + 1] = knee_angle
                joint_angles[leg_idx * 3 + 2] = ankle_angle
            
            base_x += vel_x * self.dt
            height_oscillation = 0.02 * np.sin(2 * np.pi * phase)
            base_z = self.base_height + height_oscillation
            pitch_oscillation = 0.05 * np.sin(2 * np.pi * phase)
            orientation[1] = pitch_oscillation
            
            vel = np.array([vel_x, 0.0, vel_z])
            ang_vel = np.array([0.0, 0.1 * np.cos(2 * np.pi * phase), 0.0])
            
            obs = np.concatenate([
                np.array([base_x, 0.0, base_z]),
                orientation,
                vel,
                ang_vel,
                joint_angles,
                np.random.randn(self.obs_dim - 12 - 12) * 0.01
            ])
            observations.append(obs.astype(np.float32))
            actions.append((joint_angles + np.random.randn(self.num_joints) * 0.05).astype(np.float32))
        
        return {
            'observations': observations,
            'actions': actions,
            'reward': float(np.sum([vel_x * self.dt for _ in range(num_steps)]))
        }
    
    def generate_bounding_gait(self, duration: float, phase_offset: float = 0.0) -> Dict:
        """Generate bounding gait (front/rear pairs alternate)."""
        num_steps = int(duration / self.dt)
        time = np.arange(num_steps) * self.dt
        observations = []
        actions = []
        
        base_x = 0.0
        base_z = self.base_height
        orientation = np.array([0.0, 0.0, 0.0])
        vel_x = self.stride_length * self.gait_frequency * 1.2
        
        for t in time:
            phase = (t * self.gait_frequency + phase_offset) % 1.0
            joint_angles = np.zeros(self.num_joints)
            
            for leg_idx in range(4):
                is_front = (leg_idx < 2)
                if is_front:
                    leg_phase = phase
                else:
                    leg_phase = (phase + 0.5) % 1.0
                
                hip_angle = 0.4 * np.sin(2 * np.pi * leg_phase)
                knee_angle = -0.6 * np.sin(2 * np.pi * leg_phase + np.pi/2)
                ankle_angle = 0.25 * np.sin(2 * np.pi * leg_phase)
                
                joint_angles[leg_idx * 3 + 0] = hip_angle
                joint_angles[leg_idx * 3 + 1] = knee_angle
                joint_angles[leg_idx * 3 + 2] = ankle_angle
            
            base_x += vel_x * self.dt
            base_z = self.base_height + 0.03 * np.sin(2 * np.pi * phase)
            orientation[1] = 0.08 * np.sin(2 * np.pi * phase)
            
            vel = np.array([vel_x, 0.0, 0.0])
            ang_vel = np.array([0.0, 0.15 * np.cos(2 * np.pi * phase), 0.0])
            
            obs = np.concatenate([
                np.array([base_x, 0.0, base_z]),
                orientation,
                vel,
                ang_vel,
                joint_angles,
                np.random.randn(self.obs_dim - 12 - 12) * 0.01
            ])
            observations.append(obs.astype(np.float32))
            actions.append((joint_angles + np.random.randn(self.num_joints) * 0.05).astype(np.float32))
        
        return {
            'observations': observations,
            'actions': actions,
            'reward': float(np.sum([vel_x * self.dt for _ in range(num_steps)]))
        }
    
    def generate_walking_gait(self, duration: float, phase_offset: float = 0.0) -> Dict:
        """Generate walking gait (sequential leg movement)."""
        num_steps = int(duration / self.dt)
        time = np.arange(num_steps) * self.dt
        observations = []
        actions = []
        
        base_x = 0.0
        base_z = self.base_height
        orientation = np.array([0.0, 0.0, 0.0])
        vel_x = self.stride_length * self.gait_frequency * 0.7
        
        for t in time:
            phase = (t * self.gait_frequency + phase_offset) % 1.0
            joint_angles = np.zeros(self.num_joints)
            
            for leg_idx in range(4):
                local_phase = (phase - leg_idx * 0.25) % 1.0
                
                if local_phase < 0.3:
                    swing_progress = local_phase / 0.3
                    hip_angle = 0.35 * np.sin(np.pi * swing_progress)
                    knee_angle = -0.55 * np.sin(2 * np.pi * swing_progress)
                    ankle_angle = 0.2 * np.cos(np.pi * swing_progress)
                else:
                    stance_progress = (local_phase - 0.3) / 0.7
                    hip_angle = -0.25 * np.sin(np.pi * stance_progress)
                    knee_angle = 0.35 * (1 - stance_progress)
                    ankle_angle = -0.1 * np.sin(np.pi * stance_progress)
                
                joint_angles[leg_idx * 3 + 0] = hip_angle
                joint_angles[leg_idx * 3 + 1] = knee_angle
                joint_angles[leg_idx * 3 + 2] = ankle_angle
            
            base_x += vel_x * self.dt
            base_z = self.base_height + 0.015 * np.sin(2 * np.pi * phase)
            orientation[1] = 0.03 * np.sin(2 * np.pi * phase)
            
            vel = np.array([vel_x, 0.0, 0.0])
            ang_vel = np.array([0.0, 0.08 * np.cos(2 * np.pi * phase), 0.0])
            
            obs = np.concatenate([
                np.array([base_x, 0.0, base_z]),
                orientation,
                vel,
                ang_vel,
                joint_angles,
                np.random.randn(self.obs_dim - 12 - 12) * 0.01
            ])
            observations.append(obs.astype(np.float32))
            actions.append((joint_angles + np.random.randn(self.num_joints) * 0.05).astype(np.float32))
        
        return {
            'observations': observations,
            'actions': actions,
            'reward': float(np.sum([vel_x * self.dt for _ in range(num_steps)]))
        }


def generate_realistic_demonstrations(
    output_path: str = 'data/demonstrations/expert_demos.pkl',
    num_episodes: int = 100,
    episode_duration: float = 10.0
) -> List[Dict]:
    """Generate realistic demonstration dataset."""
    
    print("=" * 70)
    print("GENERATING REALISTIC QUADRUPED DEMONSTRATION DATA")
    print("=" * 70)
    
    generator = QuadrupedGaitGenerator(num_joints=12, obs_dim=37)
    demonstrations = []
    
    trotting_episodes = int(num_episodes * 0.5)
    bounding_episodes = int(num_episodes * 0.3)
    walking_episodes = int(num_episodes * 0.2)
    
    print(f"\n📍 Generating {trotting_episodes} trotting gait episodes...")
    for i in range(trotting_episodes):
        phase_offset = np.random.uniform(0, 1.0)
        episode = generator.generate_trotting_gait(duration=episode_duration, phase_offset=phase_offset)
        demonstrations.append(episode)
        if (i + 1) % 10 == 0:
            print(f"   ✓ Generated {i + 1}/{trotting_episodes} trotting episodes")
    
    print(f"\n📍 Generating {bounding_episodes} bounding gait episodes...")
    for i in range(bounding_episodes):
        phase_offset = np.random.uniform(0, 1.0)
        episode = generator.generate_bounding_gait(duration=episode_duration, phase_offset=phase_offset)
        demonstrations.append(episode)
        if (i + 1) % 10 == 0:
            print(f"   ✓ Generated {i + 1}/{bounding_episodes} bounding episodes")
    
    print(f"\n📍 Generating {walking_episodes} walking gait episodes...")
    for i in range(walking_episodes):
        phase_offset = np.random.uniform(0, 1.0)
        episode = generator.generate_walking_gait(duration=episode_duration, phase_offset=phase_offset)
        demonstrations.append(episode)
        if (i + 1) % 10 == 0:
            print(f"   ✓ Generated {i + 1}/{walking_episodes} walking episodes")
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving {len(demonstrations)} episodes to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(demonstrations, f)
    
    print("✅ Successfully saved demonstrations!")
    
    # Statistics
    print("\n" + "=" * 70)
    print("DEMONSTRATION DATASET STATISTICS")
    print("=" * 70)
    
    all_obs = []
    all_actions = []
    total_transitions = 0
    
    for episode in demonstrations:
        all_obs.extend(episode['observations'])
        all_actions.extend(episode['actions'])
        total_transitions += len(episode['observations'])
    
    all_obs = np.array(all_obs)
    all_actions = np.array(all_actions)
    
    print(f"\nDataset Size:")
    print(f"  • Total episodes:       {len(demonstrations)}")
    print(f"  • Episode duration:     {episode_duration} seconds")
    print(f"  • Total transitions:    {total_transitions:,}")
    
    print(f"\nGait Distribution:")
    print(f"  • Trotting (50%):       {trotting_episodes} episodes")
    print(f"  • Bounding (30%):       {bounding_episodes} episodes")
    print(f"  • Walking (20%):        {walking_episodes} episodes")
    
    print(f"\nObservation Space: {all_obs.shape}")
    print(f"Action Space: {all_actions.shape}")
    
    rewards = [ep['reward'] for ep in demonstrations]
    print(f"\nReward Statistics:")
    print(f"  • Mean: {np.mean(rewards):.2f} meters")
    print(f"  • Std:  {np.std(rewards):.2f}")
    
    print("\n" + "=" * 70)
    
    return demonstrations


if __name__ == '__main__':
    demonstrations = generate_realistic_demonstrations(
        output_path='data/demonstrations/expert_demos.pkl',
        num_episodes=100,
        episode_duration=10.0
    )
    print(f"\n✅ Generated {len(demonstrations)} expert demonstration episodes!")
