"""
Training script for Behavior Cloning (BC) stage.
Pre-trains the policy using expert demonstrations.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
from stable_baselines3.common.evaluation import evaluate_policy

from src.envs.quadruped_env import QuadrupedEnv
from src.utils.data_loader import load_demonstrations, extract_transitions


class BCDataset(Dataset):
    """Dataset for Behavior Cloning."""
    
    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """Simple BC policy network."""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super(BCPolicy, self).__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Actions in [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.network(obs)


def parse_args():
    parser = argparse.ArgumentParser(description='Train BC model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--demo-path', type=str, required=True,
                        help='Path to expert demonstrations')
    parser.add_argument('--output-dir', type=str, default='models/bc_baseline',
                        help='Output directory for model checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu/cuda/auto)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for obs, actions in dataloader:
        obs = obs.to(device)
        actions = actions.to(device)
        
        # Forward pass
        predicted_actions = model(obs)
        loss = nn.MSELoss()(predicted_actions, actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for obs, actions in dataloader:
            obs = obs.to(device)
            actions = actions.to(device)
            
            predicted_actions = model(obs)
            loss = nn.MSELoss()(predicted_actions, actions)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


class BCPolicyWrapper:
    """Wrapper to make BC policy compatible with SB3 evaluation."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        with torch.no_grad():
            # Handle both single observations and batches
            if len(obs.shape) == 1:
                obs = obs.reshape(1, -1)
            
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            action = self.model(obs_tensor).cpu().numpy()
            
            # Return single action if input was single observation
            if action.shape[0] == 1:
                action = action[0]
            
        return action, state


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("\n" + "="*60)
    print("LOADING DEMONSTRATION DATA")
    print("="*60)
    
    # Load expert demonstrations
    print(f"Loading demonstrations from {args.demo_path}...")
    demonstrations = load_demonstrations(args.demo_path)
    print(f"✓ Loaded {len(demonstrations)} demonstration episodes")
    
    # Extract observations and actions
    observations, actions = extract_transitions(demonstrations)
    print(f"✓ Total transitions: {len(observations)}")
    print(f"✓ Observation shape: {observations.shape}")
    print(f"✓ Action shape: {actions.shape}")
    
    # Split into train and validation
    val_split = config.get('data_config', {}).get('validation_split', 0.1)
    n_val = int(len(observations) * val_split)
    n_train = len(observations) - n_val
    
    indices = np.random.permutation(len(observations))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_obs = observations[train_indices]
    train_actions = actions[train_indices]
    val_obs = observations[val_indices]
    val_actions = actions[val_indices]
    
    print(f"✓ Training samples: {len(train_obs)}")
    print(f"✓ Validation samples: {len(val_obs)}")
    
    # Create datasets and dataloaders
    train_dataset = BCDataset(train_obs, train_actions)
    val_dataset = BCDataset(val_obs, val_actions)
    
    batch_size = config.get('batch_size', 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n" + "="*60)
    print("CREATING BC POLICY")
    print("="*60)
    
    # Create BC policy
    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]
    hidden_dims = config.get('bc_params', {}).get('policy_kwargs', {}).get('net_arch', [256, 256])
    
    print(f"✓ Observation dim: {obs_dim}")
    print(f"✓ Action dim: {action_dim}")
    print(f"✓ Hidden layers: {hidden_dims}")
    
    model = BCPolicy(obs_dim, action_dim, hidden_dims).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total trainable parameters: {n_params:,}")
    
    # Create optimizer
    lr = config.get('bc_params', {}).get('learning_rate', 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Training loop
    n_epochs = config.get('n_epochs', 100)
    best_val_loss = float('inf')
    patience = config.get('early_stopping', {}).get('patience', 10)
    min_delta = config.get('early_stopping', {}).get('min_delta', 0.01)
    epochs_no_improve = 0
    
    print(f"Total epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Early stopping patience: {patience}")
    print()
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        if (epoch + 1) % config.get('log_interval', 10) == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if config.get('early_stopping', {}).get('enabled', True):
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_loss:.6f}")
    
    # Load best model for evaluation
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save final model
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, output_dir / 'final_model.pth')
    
    # Save training history
    np.save(output_dir / 'training_history.npy', history)
    
    print("\n" + "="*60)
    print("EVALUATING IN ENVIRONMENT")
    print("="*60)
    
    # Evaluate on environment
    env = QuadrupedEnv(render=False, **config.get('env_params', {}))
    
    policy_wrapper = BCPolicyWrapper(model, device)
    mean_reward, std_reward = evaluate_policy(
        policy_wrapper,
        env,
        n_eval_episodes=config.get('n_eval_episodes', 10),
        deterministic=True
    )
    
    print(f"✓ Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Save results
    results = {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(train_loss),
        'n_epochs_trained': len(history['train_loss']),
        'config': config,
        'seed': args.seed
    }
    
    with open(output_dir / 'results.yaml', 'w') as f:
        yaml.dump(results, f)
    
    print(f"\n✓ Model saved to: {output_dir / 'best_model.pth'}")
    print(f"✓ Results saved to: {output_dir / 'results.yaml'}")
    print("\n" + "="*60)
    print("BC TRAINING COMPLETE!")
    print("="*60)
    
    env.close()


if __name__ == '__main__':
    main()
