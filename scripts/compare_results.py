"""
Comprehensive results comparison and visualization tool.
Compares BC baseline, PPO baseline, and PPO balanced reward methods.
"""

import argparse
import json
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch

from src.envs.quadruped_env import QuadrupedEnv


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_results(results_dir):
    """Load results from yaml file."""
    results_path = Path(results_dir) / 'results.yaml'
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            # If safe_load fails due to Python objects, use unsafe load
            f.seek(0)  # Reset file pointer
            results = yaml.unsafe_load(f)
            # Clean up the config to remove non-serializable objects
            if 'config' in results:
                results['config'] = clean_config(results['config'])
            return results


def clean_config(config):
    """Remove non-serializable objects from config."""
    if isinstance(config, dict):
        cleaned = {}
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                cleaned[key] = clean_config(value)
            elif isinstance(value, str) or isinstance(value, (int, float, bool)) or value is None:
                cleaned[key] = value
            else:
                # Skip non-serializable objects
                cleaned[key] = str(type(value).__name__)
        return cleaned
    elif isinstance(config, list):
        return [clean_config(item) for item in config]
    else:
        return config

    return None


def evaluate_model(model_path, vec_normalize_path=None, n_episodes=50):
    """Evaluate a trained model."""
    print(f"\nEvaluating: {model_path}")
    
    # Create environment
    env = DummyVecEnv([lambda: Monitor(QuadrupedEnv(render=False))])
    
    # Load normalization if available
    if vec_normalize_path and Path(vec_normalize_path).exists():
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Load model
    model = PPO.load(model_path, env=env)
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if done[0]:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (i + 1) % 10 == 0:
            print(f"  Episode {i+1}/{n_episodes} - Reward: {episode_reward:.2f}")
    
    env.close()
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
    }


def create_comparison_plots(results_dict, output_dir):
    """Create comprehensive comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = list(results_dict.keys())
    
    # Figure 1: Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Methods Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean Rewards
    ax = axes[0, 0]
    mean_rewards = [results_dict[m]['mean_reward'] for m in methods]
    std_rewards = [results_dict[m]['std_reward'] for m in methods]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(methods, mean_rewards, yerr=std_rewards, capsize=10, 
                   color=colors[:len(methods)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('Episode Rewards Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, mean_rewards, std_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}\n±{std:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Episode Lengths
    ax = axes[0, 1]
    mean_lengths = [results_dict[m].get('mean_length', 0) for m in methods]
    std_lengths = [results_dict[m].get('std_length', 0) for m in methods]
    
    bars = ax.bar(methods, mean_lengths, yerr=std_lengths, capsize=10,
                   color=colors[:len(methods)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Mean Episode Length', fontsize=12, fontweight='bold')
    ax.set_title('Episode Length Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Reward Distributions
    ax = axes[1, 0]
    reward_data = [results_dict[m].get('rewards', []) for m in methods]
    
    bp = ax.boxplot(reward_data, labels=methods, patch_artist=True,
                     showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Reward Distribution', fontsize=12, fontweight='bold')
    ax.set_title('Reward Distribution Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Success Statistics
    ax = axes[1, 1]
    
    # Calculate success metrics (episodes > threshold)
    threshold = np.mean([results_dict[m]['mean_reward'] for m in methods])
    success_rates = []
    
    for method in methods:
        rewards = results_dict[method].get('rewards', [])
        if rewards:
            success_rate = 100 * np.sum(np.array(rewards) > threshold) / len(rewards)
            success_rates.append(success_rate)
        else:
            success_rates.append(0)
    
    bars = ax.bar(methods, success_rates, color=colors[:len(methods)], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Success Rate (Reward > {threshold:.1f})', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_summary.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_dir / 'comparison_summary.png'}")
    plt.close()
    
    # Figure 2: Detailed Reward Histograms
    fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
    if len(methods) == 1:
        axes = [axes]
    
    fig.suptitle('Reward Distribution Details', fontsize=16, fontweight='bold')
    
    for ax, method, color in zip(axes, methods, colors):
        rewards = results_dict[method].get('rewards', [])
        if rewards:
            ax.hist(rewards, bins=30, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label='Median')
            ax.set_xlabel('Reward', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(method, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution plot: {output_dir / 'reward_distributions.png'}")
    plt.close()


def create_summary_report(results_dict, output_path):
    """Create a text summary report."""
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("QUADRUPED LOCOMOTION TRAINING - RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        for method, results in results_dict.items():
            f.write(f"\n{method}\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Mean Reward:         {results['mean_reward']:8.2f} ± {results['std_reward']:.2f}\n")
            f.write(f"  Mean Episode Length: {results.get('mean_length', 0):8.1f} ± {results.get('std_length', 0):.1f}\n")
            
            if 'rewards' in results:
                rewards = results['rewards']
                f.write(f"  Min Reward:          {np.min(rewards):8.2f}\n")
                f.write(f"  Max Reward:          {np.max(rewards):8.2f}\n")
                f.write(f"  Median Reward:       {np.median(rewards):8.2f}\n")
            f.write("\n")
        
        # Comparison section
        if len(results_dict) > 1:
            f.write("\n" + "="*70 + "\n")
            f.write("COMPARATIVE ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            methods = list(results_dict.keys())
            baseline = methods[0]
            
            for method in methods[1:]:
                improvement = results_dict[method]['mean_reward'] - results_dict[baseline]['mean_reward']
                pct_improvement = 100 * improvement / abs(results_dict[baseline]['mean_reward'])
                
                f.write(f"{method} vs {baseline}:\n")
                f.write(f"  Absolute improvement: {improvement:+.2f}\n")
                f.write(f"  Relative improvement: {pct_improvement:+.2f}%\n\n")
    
    print(f"✓ Saved summary report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare training results')
    parser.add_argument('--bc-dir', type=str, default='models/bc_baseline',
                        help='BC model directory')
    parser.add_argument('--ppo-baseline-dir', type=str, default='models/ppo_baseline',
                        help='PPO baseline model directory')
    parser.add_argument('--ppo-balanced-dir', type=str, default='models/ppo_balanced',
                        help='PPO balanced model directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--n-eval-episodes', type=int, default=50,
                        help='Number of evaluation episodes')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip model evaluation, use existing results only')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*70)
    
    results_dict = {}
    
    # Load or evaluate BC model
    if Path(args.bc_dir).exists():
        print(f"\n📊 BC Baseline ({args.bc_dir})")
        bc_results = load_results(args.bc_dir)
        if bc_results:
            results_dict['BC Baseline'] = bc_results
            print(f"  Mean reward: {bc_results['mean_reward']:.2f} ± {bc_results['std_reward']:.2f}")
    
    # Load or evaluate PPO baseline
    if Path(args.ppo_baseline_dir).exists():
        print(f"\n📊 PPO Baseline ({args.ppo_baseline_dir})")
        
        if not args.skip_eval:
            model_path = Path(args.ppo_baseline_dir) / 'best_model.zip'
            if not model_path.exists():
                model_path = Path(args.ppo_baseline_dir) / 'final_model.zip'
            
            if model_path.exists():
                vec_norm_path = Path(args.ppo_baseline_dir) / 'vec_normalize.pkl'
                ppo_results = evaluate_model(model_path, vec_norm_path, args.n_eval_episodes)
                results_dict['PPO Baseline'] = ppo_results
                print(f"  Mean reward: {ppo_results['mean_reward']:.2f} ± {ppo_results['std_reward']:.2f}")
        else:
            ppo_results = load_results(args.ppo_baseline_dir)
            if ppo_results:
                results_dict['PPO Baseline'] = ppo_results
    
    # Load or evaluate PPO balanced
    if Path(args.ppo_balanced_dir).exists():
        print(f"\n📊 PPO Balanced ({args.ppo_balanced_dir})")
        
        if not args.skip_eval:
            model_path = Path(args.ppo_balanced_dir) / 'best_model.zip'
            if not model_path.exists():
                model_path = Path(args.ppo_balanced_dir) / 'final_model.zip'
            
            if model_path.exists():
                vec_norm_path = Path(args.ppo_balanced_dir) / 'vec_normalize.pkl'
                balanced_results = evaluate_model(model_path, vec_norm_path, args.n_eval_episodes)
                results_dict['PPO Balanced'] = balanced_results
                print(f"  Mean reward: {balanced_results['mean_reward']:.2f} ± {balanced_results['std_reward']:.2f}")
        else:
            balanced_results = load_results(args.ppo_balanced_dir)
            if balanced_results:
                results_dict['PPO Balanced'] = balanced_results
    
    if not results_dict:
        print("\n❌ No results found to compare!")
        return
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_comparison_plots(results_dict, output_dir)
    create_summary_report(results_dict, output_dir / 'summary_report.txt')
    
    # Save detailed results as JSON
    detailed_results = {
        method: {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                for k, v in results.items()}
        for method, results in results_dict.items()
    }
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"✓ Saved detailed results: {output_dir / 'detailed_results.json'}")
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - comparison_summary.png")
    print(f"  - reward_distributions.png")
    print(f"  - summary_report.txt")
    print(f"  - detailed_results.json")


if __name__ == '__main__':
    main()
