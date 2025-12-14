"""
Demo script for visualizing trained policy.
Loads a trained model and runs it in the environment.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.quadruped_env import QuadrupedEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Run demo with trained model')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--vec-normalize', type=str, default=None,
                        help='Path to VecNormalize stats')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    parser.add_argument('--record', action='store_true',
                        help='Record video')
    parser.add_argument('--output', type=str, default='demo_video.mp4',
                        help='Output video filename')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic actions')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Delay between steps (seconds)')
    return parser.parse_args()


def make_env(render=False):
    """Create environment."""
    def _init():
        env = QuadrupedEnv(render=render)
        return env
    return _init


def main():
    args = parse_args()
    
    print(f"Loading model from {args.weights}")
    
    # Create environment
    env = DummyVecEnv([make_env(render=args.render)])
    
    # Load normalization stats if provided
    if args.vec_normalize:
        print(f"Loading VecNormalize stats from {args.vec_normalize}")
        env = VecNormalize.load(args.vec_normalize, env)
        env.training = False
        env.norm_reward = False
    
    # Load trained model
    model = PPO.load(args.weights, env=env)
    print("Model loaded successfully")
    
    # Initialize video recording
    frames = [] if args.record else None
    
    # Run episodes
    total_rewards = []
    total_steps = []
    
    for episode in range(args.episodes):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=args.deterministic)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_steps += 1
            
            # Record frame
            if args.record:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)
            
            # Add delay if specified
            if args.delay > 0:
                time.sleep(args.delay)
            
            # Check if episode is done
            if done[0]:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        
        print(f"Episode {episode + 1} finished")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {episode_steps}")
        
        # Print info if available
        if info and len(info) > 0:
            info_dict = info[0]
            if 'episode' in info_dict:
                print(f"  Episode info: {info_dict['episode']}")
    
    # Print statistics
    print("\n=== Summary Statistics ===")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Mean steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    
    # Save video if recording
    if args.record and frames:
        print(f"\nSaving video to {args.output}")
        imageio.mimsave(args.output, frames, fps=args.fps)
        print(f"Video saved with {len(frames)} frames")
    
    env.close()
    print("\nDemo complete!")


if __name__ == '__main__':
    main()
