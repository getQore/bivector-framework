#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lambda-PPO vs Standard PPO Comparison
======================================

Compare Lambda-PPO (with bivector weighting) vs standard PPO on CartPole.

Day 4: RL Sprint - Atomic Physics Application
Rick Mathews - November 14, 2024
"""

import numpy as np
import torch
import gymnasium as gym
from lambda_ppo_starter import LambdaPPO, collect_trajectories


def run_experiment(use_lambda, n_runs=5, max_iterations=100):
    """
    Run PPO with or without Lambda weighting.

    Args:
        use_lambda: Whether to use Lambda weighting
        n_runs: Number of independent runs
        max_iterations: Maximum training iterations

    Returns:
        Dictionary of results
    """
    convergence_iterations = []
    final_rewards = []
    lambda_values = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)

        env = gym.make('CartPole-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Create agent
        agent = LambdaPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            use_lambda=use_lambda,
            lr_policy=3e-4,
            lr_value=1e-3
        )

        converged = False
        convergence_iter = max_iterations
        run_lambda_values = []

        for iteration in range(max_iterations):
            # Collect trajectories
            trajectories = collect_trajectories(env, agent, n_steps=2048)

            # Update
            metrics = agent.update(
                states=trajectories['states'],
                actions=trajectories['actions'],
                rewards=trajectories['rewards'],
                dones=trajectories['dones'],
                old_log_probs=trajectories['log_probs'],
                old_logits=trajectories['logits'],
                epochs=10
            )

            run_lambda_values.append(metrics['mean_lambda'])

            # Check convergence
            if len(trajectories['episode_rewards']) > 0:
                avg_reward = np.mean(trajectories['episode_rewards'])

                if avg_reward > 195 and not converged:
                    convergence_iter = iteration + 1
                    converged = True
                    break

        env.close()

        convergence_iterations.append(convergence_iter)
        if len(trajectories['episode_rewards']) > 0:
            final_rewards.append(np.mean(trajectories['episode_rewards']))
        else:
            final_rewards.append(0)
        lambda_values.append(np.mean(run_lambda_values))

        print(f"Conv={convergence_iter}, Lambda={np.mean(run_lambda_values):.6f}")

    return {
        'convergence_mean': np.mean(convergence_iterations),
        'convergence_std': np.std(convergence_iterations),
        'final_reward_mean': np.mean(final_rewards),
        'final_reward_std': np.std(final_rewards),
        'lambda_mean': np.mean(lambda_values),
        'lambda_std': np.std(lambda_values),
        'all_convergence': convergence_iterations
    }


def main():
    """Main comparison"""
    print("\n")
    print("="*80)
    print("LAMBDA-PPO vs STANDARD PPO COMPARISON")
    print("="*80)
    print()
    print("Task: CartPole-v1 (solved at avg reward > 195)")
    print("Runs: 5 independent trials per algorithm")
    print()

    # Standard PPO
    print("Testing STANDARD PPO (no Lambda weighting)...")
    standard_results = run_experiment(use_lambda=False, n_runs=5)

    print()

    # Lambda-PPO
    print("Testing LAMBDA-PPO (with bivector weighting)...")
    lambda_results = run_experiment(use_lambda=True, n_runs=5)

    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()

    print(f"{'Algorithm':<30} {'Convergence (iter)':<20} {'Final Reward':<15} {'Mean Lambda'}")
    print("-"*80)

    print(f"{'Standard PPO':<30} "
          f"{standard_results['convergence_mean']:.1f} ± {standard_results['convergence_std']:.1f}"
          f"{'':<6} "
          f"{standard_results['final_reward_mean']:.1f} ± {standard_results['final_reward_std']:.1f}"
          f"{'':<2} "
          f"{standard_results['lambda_mean']:.6f}")

    print(f"{'Lambda-PPO':<30} "
          f"{lambda_results['convergence_mean']:.1f} ± {lambda_results['convergence_std']:.1f}"
          f"{'':<6} "
          f"{lambda_results['final_reward_mean']:.1f} ± {lambda_results['final_reward_std']:.1f}"
          f"{'':<2} "
          f"{lambda_results['lambda_mean']:.6f}")

    print()

    # Statistical comparison
    improvement = 100 * (1 - lambda_results['convergence_mean'] / standard_results['convergence_mean'])

    print(f"Lambda-PPO vs Standard PPO:")
    print(f"  Convergence: {improvement:+.1f}% {'faster' if improvement > 0 else 'slower'}")
    print()

    # Analysis
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()

    if lambda_results['lambda_mean'] < 0.001:
        print("⚠️  CRITICAL FINDING: Lambda values extremely small (< 0.001)")
        print()
        print("Possible explanations:")
        print("1. Discrete action distributions have low commutator norms")
        print("2. PPO's smooth policy updates → small Λ between old/new policies")
        print("3. Bivector encoding may not capture discrete distribution differences")
        print()
        print("Implications for patent:")
        print("- Lambda-PPO implementation works (solves CartPole)")
        print("- BUT: Lambda weighting may not provide measurable advantage")
        print("- Recomm: Focus patent claims on multi-armed bandits (validated)")
        print("- Lambda-PPO needs different environments or encoding method")
    else:
        print(f"✅ Lambda values detected: {lambda_results['lambda_mean']:.6f}")
        if improvement > 10:
            print(f"✅ Lambda-PPO shows significant improvement: {improvement:+.1f}%")
        elif improvement > 0:
            print(f"⚠️  Lambda-PPO shows marginal improvement: {improvement:+.1f}%")
        else:
            print(f"❌ Lambda-PPO slower than standard PPO: {improvement:+.1f}%")

    print()
    print("="*80)
    print()

    # Save results
    results = {
        'standard_ppo': standard_results,
        'lambda_ppo': lambda_results,
        'improvement_pct': improvement
    }

    import json
    with open('lambda_ppo_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: lambda_ppo_comparison_results.json")
    print()


if __name__ == "__main__":
    main()
