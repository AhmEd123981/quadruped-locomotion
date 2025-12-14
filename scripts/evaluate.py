
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from src.envs.quadruped_env import QuadrupedEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--vec-normalize', type=str, default=None,
                        help='Path to VecNormalize stats')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default='results/evaluation_metrics.json',
                        help='Output file for results')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic actions')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for evaluation')
    return parser.parse_args()


def make_env(seed=0):
    """Create environment."""
    def _init():
        env = QuadrupedEnv(render=False)
        env.seed(seed)
        return env
    return _init


def compute_metrics(episode_data):
    """Compute detailed metrics from episode data."""
    metrics = {}
    
    # Basic statistics
    rewards = [ep['reward'] for ep in episode_data]
    steps = [ep['steps'] for ep in episode_data]
    success = [ep.get('success', False) for ep in episode_data]
    
    metrics['reward'] = {
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
        'min': float(np.min(rewards)),
        'max': float(np.max(rewards)),
        'median': float(np.median(rewards))
    }
    
    metrics['steps'] = {
        'mean': float(np.mean(steps)),
        'std': float(np.std(steps)),
        'min': int(np.min(steps)),
        'max': int(np.max(steps))
    }
    
    metrics['success_rate'] = float(np.mean(success))
    
    # Additional metrics if available
    if 'speed' in episode_data[0]:
        speeds = [ep['speed'] for ep in episode_data]
        metrics['speed'] = {
            'mean': float(np.mean(speeds)),
            'std': float(np.std(speeds)),
            'min': float(np.min(speeds)),
            'max': float(np.max(speeds))
        }
    
    if 'gait_stability' in episode_data[0]:
        stabilities = [ep['gait_stability'] for ep in episode_data]
        metrics['gait_stability'] = {
            'mean': float(np.mean(stabilities)),
            'std': float(np.std(stabilities))
        }
    
    if 'falls' in episode_data[0]:
        falls = [ep['falls'] for ep in episode_data]
        metrics['falls'] = {
            'total': int(np.sum(falls)),
            'mean_per_episode': float(np.mean(falls)),
            'episodes_with_falls': int(np.sum(np.array(falls) > 0))
        }
    
    return metrics


def main():
    args = parse_args()
    
    print(f"Evaluating model: {args.weights}")
    print(f"Number of episodes: {args.num_episodes}")
    
    # Create environment
    env = DummyVecEnv([make_env(seed=args.seed)])
    
    # Load normalization stats if provided
    if args.vec_normalize:
        print(f"Loading VecNormalize stats from {args.vec_normalize}")
        env = VecNormalize.load(args.vec_normalize, env)
        env.training = False
        env.norm_reward = False
    
    # Load trained model
    print("Loading model...")
    model = PPO.load(args.weights, env=env)
    
    # Run evaluation episodes
    print("Running evaluation...")
    episode_data = []
    
    for episode in tqdm(range(args.num_episodes)):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Episode-specific metrics
        episode_info = {
            'episode': episode,
            'reward': 0,
            'steps': 0,
            'success': False
        }
        
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_steps += 1
            
            if done[0]:
                break
        
        # Store episode data
        episode_info['reward'] = float(episode_reward)
        episode_info['steps'] = int(episode_steps)
        
        # Extract additional info if available
        if info and len(info) > 0:
            info_dict = info[0]
            if 'success' in info_dict:
                episode_info['success'] = bool(info_dict['success'])
            if 'speed' in info_dict:
                episode_info['speed'] = float(info_dict['speed'])
            if 'gait_stability' in info_dict:
                episode_info['gait_stability'] = float(info_dict['gait_stability'])
            if 'falls' in info_dict:
                episode_info['falls'] = int(info_dict['falls'])
        
        episode_data.append(episode_info)
    
    env.close()
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(episode_data)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nReward Statistics:")
    print(f"  Mean: {metrics['reward']['mean']:.2f} ± {metrics['reward']['std']:.2f}")
    print(f"  Min:  {metrics['reward']['min']:.2f}")
    print(f"  Max:  {metrics['reward']['max']:.2f}")
    print(f"  Median: {metrics['reward']['median']:.2f}")
    
    print(f"\nSteps Statistics:")
    print(f"  Mean: {metrics['steps']['mean']:.1f} ± {metrics['steps']['std']:.1f}")
    print(f"  Min:  {metrics['steps']['min']}")
    print(f"  Max:  {metrics['steps']['max']}")
    
    print(f"\nSuccess Rate: {metrics['success_rate']*100:.1f}%")
    
    if 'speed' in metrics:
        print(f"\nSpeed Statistics:")
        print(f"  Mean: {metrics['speed']['mean']:.2f} ± {metrics['speed']['std']:.2f} m/s")
        print(f"  Min:  {metrics['speed']['min']:.2f} m/s")
        print(f"  Max:  {metrics['speed']['max']:.2f} m/s")
    
    if 'gait_stability' in metrics:
        print(f"\nGait Stability: {metrics['gait_stability']['mean']:.3f} ± {metrics['gait_stability']['std']:.3f}")
    
    if 'falls' in metrics:
        print(f"\nFall Statistics:")
        print(f"  Total falls: {metrics['falls']['total']}")
        print(f"  Mean per episode: {metrics['falls']['mean_per_episode']:.2f}")
        print(f"  Episodes with falls: {metrics['falls']['episodes_with_falls']}/{args.num_episodes}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_path': args.weights,
        'num_episodes': args.num_episodes,
        'deterministic': args.deterministic,
        'seed': args.seed,
        'metrics': metrics,
        'episode_data': episode_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
