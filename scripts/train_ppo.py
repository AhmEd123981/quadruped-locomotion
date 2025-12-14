"""
Training script for PPO fine-tuning stage.
Fine-tunes BC pre-trained policy using PPO with custom reward function.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.quadruped_env import QuadrupedEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--bc-checkpoint', type=str, default=None,
                        help='Path to BC checkpoint for initialization')
    parser.add_argument('--output-dir', type=str, default='models/ppo_finetuned',
                        help='Output directory for model checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_env(config, seed, rank=0):
    """Create and wrap environment."""
    def _init():
        # Extract reward config
        reward_config = config.get('reward_config', {})
        env_params = config.get('env_params', {}).copy()
        
        # Add reward weights to env params
        if 'speed_weight' not in env_params:
            env_params['speed_weight'] = reward_config.get('speed_weight', 0.6)
        if 'gait_weight' not in env_params:
            env_params['gait_weight'] = reward_config.get('gait_weight', 0.4)
        
        # Set render
        env_params['render'] = config.get('render', False)
        
        env = QuadrupedEnv(**env_params)
        env.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print("="*60)
    print("PPO FINE-TUNING CONFIGURATION")
    print("="*60)
    print(f"Config file: {args.config}")
    print(f"Reward type: {config.get('env_params', {}).get('reward_type', 'balanced')}")
    print(f"Output directory: {args.output_dir}")
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("\n" + "="*60)
    print("CREATING TRAINING ENVIRONMENTS")
    print("="*60)
    
    # Create vectorized environments
    n_envs = config.get('n_envs', 4)
    print(f"Number of parallel environments: {n_envs}")
    
    env = DummyVecEnv([make_env(config, args.seed, i) for i in range(n_envs)])
    
    # Normalize observations and rewards
    if config.get('normalize', True):
        print("✓ Applying VecNormalize wrapper")
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(config, args.seed + 1000, 0)])
    if config.get('normalize', True):
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=True,
            norm_reward=False
        )
    
    print("\n" + "="*60)
    print("INITIALIZING PPO MODEL")
    print("="*60)
    
    # Get PPO hyperparameters
    ppo_params = config.get('ppo_params', {})
    
    # Convert activation function string to actual function
    policy_kwargs = ppo_params.get('policy_kwargs', {}).copy()
    if 'activation_fn' in policy_kwargs:
        activation_fn_str = policy_kwargs['activation_fn']
        if activation_fn_str == 'relu':
            policy_kwargs['activation_fn'] = nn.ReLU
        elif activation_fn_str == 'tanh':
            policy_kwargs['activation_fn'] = nn.Tanh
        elif activation_fn_str == 'elu':
            policy_kwargs['activation_fn'] = nn.ELU
        else:
            policy_kwargs['activation_fn'] = nn.ReLU
    ppo_params['policy_kwargs'] = policy_kwargs
    
    # Initialize or load PPO model
    if args.resume:
        print(f"Resuming training from {args.resume}")
        model = PPO.load(
            args.resume,
            env=env,
            device=args.device
        )
    elif args.bc_checkpoint:
        print(f"Initializing from BC checkpoint: {args.bc_checkpoint}")
        
        # Create PPO model
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=ppo_params.get('learning_rate', 3e-4),
            n_steps=ppo_params.get('n_steps', 2048),
            batch_size=ppo_params.get('batch_size', 64),
            n_epochs=ppo_params.get('n_epochs', 10),
            gamma=ppo_params.get('gamma', 0.99),
            gae_lambda=ppo_params.get('gae_lambda', 0.95),
            clip_range=ppo_params.get('clip_range', 0.2),
            ent_coef=ppo_params.get('ent_coef', 0.01),
            vf_coef=ppo_params.get('vf_coef', 0.5),
            max_grad_norm=ppo_params.get('max_grad_norm', 0.5),
            tensorboard_log=str(output_dir / 'tensorboard'),
            policy_kwargs=ppo_params.get('policy_kwargs', {}),
            verbose=ppo_params.get('verbose', 1),
            device=args.device,
            seed=args.seed
        )
        
        # Try to load BC weights into PPO policy
        try:
            bc_checkpoint = torch.load(args.bc_checkpoint, map_location=args.device)
            print("✓ BC checkpoint loaded")
            
            # Note: Direct weight transfer from BC to PPO is tricky
            # For now, we'll just train PPO from scratch
            print("⚠ Training PPO from scratch (BC weight transfer not implemented)")
        except Exception as e:
            print(f"⚠ Could not load BC weights: {e}")
            print("Training PPO from scratch")
    else:
        print("Training PPO from scratch (no BC initialization)")
        
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=ppo_params.get('learning_rate', 3e-4),
            n_steps=ppo_params.get('n_steps', 2048),
            batch_size=ppo_params.get('batch_size', 64),
            n_epochs=ppo_params.get('n_epochs', 10),
            gamma=ppo_params.get('gamma', 0.99),
            gae_lambda=ppo_params.get('gae_lambda', 0.95),
            clip_range=ppo_params.get('clip_range', 0.2),
            ent_coef=ppo_params.get('ent_coef', 0.01),
            vf_coef=ppo_params.get('vf_coef', 0.5),
            max_grad_norm=ppo_params.get('max_grad_norm', 0.5),
            tensorboard_log=str(output_dir / 'tensorboard'),
            policy_kwargs=ppo_params.get('policy_kwargs', {}),
            verbose=ppo_params.get('verbose', 1),
            device=args.device,
            seed=args.seed
        )
    
    print(f"✓ PPO model initialized")
    print(f"✓ Device: {model.device}")
    print(f"✓ Learning rate: {model.learning_rate}")
    print(f"✓ Steps per update: {model.n_steps}")
    print(f"✓ Batch size: {model.batch_size}")
    
    print("\n" + "="*60)
    print("SETTING UP CALLBACKS")
    print("="*60)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('checkpoint_freq', 50000),
        save_path=str(output_dir / 'checkpoints'),
        name_prefix='ppo_model',
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir / 'eval_logs'),
        eval_freq=config.get('eval_freq', 10000),
        n_eval_episodes=config.get('n_eval_episodes', 10),
        deterministic=True,
        render=False
    )
    
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    print("✓ Checkpoint callback configured")
    print("✓ Evaluation callback configured")
    
    print("\n" + "="*60)
    print("STARTING PPO TRAINING")
    print("="*60)
    
    total_timesteps = config.get('total_timesteps', 1000000)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Estimated updates: {total_timesteps // (n_envs * model.n_steps):,}")
    print()
    
    # Train PPO model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=config.get('log_interval', 10),
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    
    print("\n" + "="*60)
    print("SAVING FINAL MODEL")
    print("="*60)
    
    # Save final model
    final_model_path = output_dir / 'final_model.zip'
    model.save(str(final_model_path))
    print(f"✓ Model saved to: {final_model_path}")
    
    # Save VecNormalize stats
    if config.get('normalize', True):
        env.save(str(output_dir / 'vec_normalize.pkl'))
        print(f"✓ VecNormalize stats saved")
    
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Final evaluation
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True
    )
    
    print(f"✓ Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Save results
    results = {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'total_timesteps': total_timesteps,
        'config': config,
        'seed': args.seed
    }
    
    with open(output_dir / 'results.yaml', 'w') as f:
        yaml.dump(results, f)
    
    print(f"✓ Results saved to: {output_dir / 'results.yaml'}")
    
    print("\n" + "="*60)
    print("PPO TRAINING COMPLETE!")
    print("="*60)
    
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
