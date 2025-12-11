"""
Utilities for loading and processing demonstration data.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def load_demonstrations(file_path: str) -> List[Dict]:
    """
    Load expert demonstrations from file.
    
    Args:
        file_path: Path to demonstrations file (.pkl)
        
    Returns:
        List of episode dictionaries containing observations and actions
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Demonstration file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        demonstrations = pickle.load(f)
    
    # Validate demonstration format
    if not isinstance(demonstrations, list):
        raise ValueError("Demonstrations must be a list of episodes")
    
    for i, episode in enumerate(demonstrations):
        if not isinstance(episode, dict):
            raise ValueError(f"Episode {i} must be a dictionary")
        if 'observations' not in episode or 'actions' not in episode:
            raise ValueError(f"Episode {i} missing 'observations' or 'actions'")
    
    return demonstrations


def extract_transitions(
    demonstrations: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all state-action pairs from demonstrations.
    
    Args:
        demonstrations: List of episode dictionaries
        
    Returns:
        Tuple of (observations, actions) as numpy arrays
    """
    all_observations = []
    all_actions = []
    
    for episode in demonstrations:
        all_observations.extend(episode['observations'])
        all_actions.extend(episode['actions'])
    
    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    
    return observations, actions


def split_demonstrations(
    demonstrations: List[Dict],
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split demonstrations into train and validation sets.
    
    Args:
        demonstrations: List of episode dictionaries
        train_ratio: Ratio of episodes for training
        shuffle: Whether to shuffle episodes before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_demos, val_demos)
    """
    np.random.seed(random_seed)
    
    demos = demonstrations.copy()
    
    if shuffle:
        np.random.shuffle(demos)
    
    split_idx = int(len(demos) * train_ratio)
    train_demos = demos[:split_idx]
    val_demos = demos[split_idx:]
    
    return train_demos, val_demos


def compute_demonstration_statistics(
    demonstrations: List[Dict]
) -> Dict:
    """
    Compute statistics about demonstration data.
    
    Args:
        demonstrations: List of episode dictionaries
        
    Returns:
        Dictionary of statistics
    """
    num_episodes = len(demonstrations)
    
    episode_lengths = [len(ep['observations']) for ep in demonstrations]
    episode_rewards = [ep.get('reward', 0) for ep in demonstrations]
    
    # Extract all observations and actions
    observations, actions = extract_transitions(demonstrations)
    
    stats = {
        'num_episodes': num_episodes,
        'total_transitions': len(observations),
        'episode_length': {
            'mean': float(np.mean(episode_lengths)),
            'std': float(np.std(episode_lengths)),
            'min': int(np.min(episode_lengths)),
            'max': int(np.max(episode_lengths))
        },
        'episode_reward': {
            'mean': float(np.mean(episode_rewards)),
            'std': float(np.std(episode_rewards)),
            'min': float(np.min(episode_rewards)),
            'max': float(np.max(episode_rewards))
        },
        'observation': {
            'shape': observations.shape,
            'mean': observations.mean(axis=0).tolist(),
            'std': observations.std(axis=0).tolist(),
            'min': observations.min(axis=0).tolist(),
            'max': observations.max(axis=0).tolist()
        },
        'action': {
            'shape': actions.shape,
            'mean': actions.mean(axis=0).tolist(),
            'std': actions.std(axis=0).tolist(),
            'min': actions.min(axis=0).tolist(),
            'max': actions.max(axis=0).tolist()
        }
    }
    
    return stats


def normalize_demonstrations(
    demonstrations: List[Dict],
    obs_mean: np.ndarray = None,
    obs_std: np.ndarray = None
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Normalize observations in demonstrations.
    
    Args:
        demonstrations: List of episode dictionaries
        obs_mean: Pre-computed observation mean (optional)
        obs_std: Pre-computed observation std (optional)
        
    Returns:
        Tuple of (normalized_demos, obs_mean, obs_std)
    """
    # Compute normalization statistics if not provided
    if obs_mean is None or obs_std is None:
        all_obs = []
        for episode in demonstrations:
            all_obs.extend(episode['observations'])
        all_obs = np.array(all_obs)
        obs_mean = all_obs.mean(axis=0)
        obs_std = all_obs.std(axis=0) + 1e-8  # Avoid division by zero
    
    # Normalize demonstrations
    normalized_demos = []
    for episode in demonstrations:
        normalized_obs = (np.array(episode['observations']) - obs_mean) / obs_std
        normalized_episode = {
            'observations': normalized_obs.tolist(),
            'actions': episode['actions'],
            **{k: v for k, v in episode.items() if k not in ['observations', 'actions']}
        }
        normalized_demos.append(normalized_episode)
    
    return normalized_demos, obs_mean, obs_std


def save_demonstrations(
    demonstrations: List[Dict],
    file_path: str
):
    """
    Save demonstrations to file.
    
    Args:
        demonstrations: List of episode dictionaries
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(demonstrations, f)
    
    print(f"Saved {len(demonstrations)} episodes to {file_path}")


def create_dummy_demonstrations(
    num_episodes: int = 10,
    episode_length: int = 100,
    obs_dim: int = 37,
    action_dim: int = 12
) -> List[Dict]:
    """
    Create dummy demonstrations for testing.
    
    Args:
        num_episodes: Number of episodes
        episode_length: Steps per episode
        obs_dim: Observation dimension
        action_dim: Action dimension
        
    Returns:
        List of dummy episode dictionaries
    """
    demonstrations = []
    
    for _ in range(num_episodes):
        observations = np.random.randn(episode_length, obs_dim).tolist()
        actions = np.random.uniform(-1, 1, (episode_length, action_dim)).tolist()
        
        episode = {
            'observations': observations,
            'actions': actions,
            'reward': np.random.uniform(0, 100)
        }
        demonstrations.append(episode)
    
    return demonstrations
