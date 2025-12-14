"""
Visualization utilities for training analysis and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import json


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_curves(
    tensorboard_log_dir: str,
    output_path: str,
    metrics: List[str] = ['rollout/ep_rew_mean', 'train/loss']
):
    """
    Plot training curves from TensorBoard logs.
    
    Args:
        tensorboard_log_dir: Path to TensorBoard log directory
        output_path: Path to save plot
        metrics: List of metrics to plot
    """
    from tensorboard.backend.event_processing import event_accumulator
    
    log_dir = Path(tensorboard_log_dir)
    
    # Find event files
    event_files = list(log_dir.rglob('events.out.tfevents.*'))
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return
    
    # Load data
    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()
    
    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in ea.Tags()['scalars']:
            events = ea.Scalars(metric)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            axes[i].plot(steps, values, linewidth=2)
            axes[i].set_xlabel('Steps')
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric} over Training')
            axes[i].grid(True, alpha=0.3)
        else:
            print(f"Metric {metric} not found in logs")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {output_path}")


def plot_comparison(
    results: Dict[str, Dict],
    output_path: str,
    metrics: List[str] = ['reward', 'speed', 'gait_stability']
):
    """
    Plot comparison between different methods.
    
    Args:
        results: Dictionary mapping method names to their results
        output_path: Path to save plot
        metrics: List of metrics to compare
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    method_names = list(results.keys())
    colors = sns.color_palette("husl", len(method_names))
    
    for i, metric in enumerate(metrics):
        values = []
        errors = []
        
        for method in method_names:
            if metric in results[method]['metrics']:
                metric_data = results[method]['metrics'][metric]
                values.append(metric_data['mean'])
                errors.append(metric_data.get('std', 0))
            else:
                values.append(0)
                errors.append(0)
        
        x_pos = np.arange(len(method_names))
        axes[i].bar(x_pos, values, yerr=errors, capsize=5, 
                   color=colors, alpha=0.7, edgecolor='black')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(method_names, rotation=15, ha='right')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {output_path}")


def plot_episode_rewards(
    episode_data: List[Dict],
    output_path: str,
    window_size: int = 10
):
    """
    Plot episode rewards with moving average.
    
    Args:
        episode_data: List of episode dictionaries
        output_path: Path to save plot
        window_size: Window size for moving average
    """
    rewards = [ep['reward'] for ep in episode_data]
    episodes = list(range(len(rewards)))
    
    # Compute moving average
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, alpha=0.3, label='Episode Reward')
    plt.plot(episodes[window_size-1:], moving_avg, linewidth=2, 
            label=f'Moving Average (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards During Evaluation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Episode rewards plot saved to {output_path}")


def plot_action_distribution(
    actions: np.ndarray,
    output_path: str,
    joint_names: Optional[List[str]] = None
):
    """
    Plot action distribution across joints.
    
    Args:
        actions: Array of actions (num_steps, num_joints)
        output_path: Path to save plot
        joint_names: Names of joints (optional)
    """
    num_joints = actions.shape[1]
    
    if joint_names is None:
        joint_names = [f'Joint {i+1}' for i in range(num_joints)]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(num_joints):
        axes[i].hist(actions[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel('Action Value')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(joint_names[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(actions[:, i].mean(), color='red', 
                       linestyle='--', label='Mean')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Action distribution plot saved to {output_path}")


def plot_gait_analysis(
    joint_trajectories: np.ndarray,
    output_path: str,
    time_step: float = 0.01
):
    """
    Plot joint trajectories for gait analysis.
    
    Args:
        joint_trajectories: Array of joint positions (num_steps, num_joints)
        output_path: Path to save plot
        time_step: Time step between samples
    """
    num_steps, num_joints = joint_trajectories.shape
    time = np.arange(num_steps) * time_step
    
    # Plot all joints
    plt.figure(figsize=(14, 8))
    
    # Group joints by leg (assuming 3 joints per leg)
    leg_colors = ['blue', 'red', 'green', 'orange']
    leg_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']
    
    for leg_idx in range(4):
        for joint_idx in range(3):
            idx = leg_idx * 3 + joint_idx
            if idx < num_joints:
                plt.plot(time, joint_trajectories[:, idx], 
                        color=leg_colors[leg_idx],
                        alpha=0.5 + 0.5 * (joint_idx / 3),
                        label=f'{leg_names[leg_idx]} - Joint {joint_idx+1}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (rad)')
    plt.title('Joint Trajectories - Gait Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gait analysis plot saved to {output_path}")


def create_results_summary(
    results_dict: Dict,
    output_path: str
):
    """
    Create a visual summary of results.
    
    Args:
        results_dict: Dictionary containing results
        output_path: Path to save summary
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Title
    fig.suptitle('Quadruped Locomotion Training Results', 
                fontsize=16, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Key metrics
    ax1 = fig.add_subplot(gs[0, :])
    metrics = ['reward', 'speed', 'gait_stability', 'success_rate']
    values = [results_dict['metrics'][m]['mean'] for m in metrics if m in results_dict['metrics']]
    x_pos = np.arange(len(values))
    ax1.bar(x_pos, values, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics[:len(values)]])
    ax1.set_title('Key Performance Metrics')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Episode rewards
    if 'episode_data' in results_dict:
        ax2 = fig.add_subplot(gs[1, 0])
        rewards = [ep['reward'] for ep in results_dict['episode_data']]
        ax2.plot(rewards, alpha=0.5)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.set_title('Episode Rewards')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution
    if 'episode_data' in results_dict:
        ax3 = fig.add_subplot(gs[1, 1])
        rewards = [ep['reward'] for ep in results_dict['episode_data']]
        ax3.hist(rewards, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distribution')
        ax3.grid(True, alpha=0.3)
    
    # Text summary
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary_text = f"""
    Training Summary:
    ─────────────────────────────────────────────
    Total Episodes: {results_dict.get('num_episodes', 'N/A')}
    Mean Reward: {results_dict['metrics']['reward']['mean']:.2f} ± {results_dict['metrics']['reward']['std']:.2f}
    Success Rate: {results_dict['metrics'].get('success_rate', 0)*100:.1f}%
    
    Model: {results_dict.get('model_path', 'N/A')}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontfamily='monospace', 
            fontsize=11, verticalalignment='center')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results summary saved to {output_path}")
