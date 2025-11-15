#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit 3-Door Problem: Lambda-Bandit Validation
================================================

Exact problem from RL patent use case:
- 3 doors with overlapping reward distributions
- Door 1: N(μ=1, σ=0.1) - Low mean, low variance
- Door 2: N(μ=2, σ=2.0) - High mean, high variance (OPTIMAL)
- Door 3: N(μ=1.5, σ=0.5) - Medium mean, medium variance

Challenge: High overlap makes optimal door (Door 2) hard to identify

Test: Lambda-Bandit converges faster than standard methods

Rick Mathews - RL Patent Application
November 14, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from distribution_bivector_utils import (
    compute_distribution_lambda,
    gaussian_to_bivector,
    BivectorCl31
)
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================================
# DOOR DEFINITIONS (EXACT FROM REDDIT POST)
# ============================================================================

DOORS = [
    {'name': 'Door 1', 'mu': 1.0, 'sigma': 0.1},   # Consistent but weak
    {'name': 'Door 2', 'mu': 2.0, 'sigma': 2.0},   # Risky but best
    {'name': 'Door 3', 'mu': 1.5, 'sigma': 0.5}    # Balanced
]

OPTIMAL_ARM = 1  # Door 2 has highest mean


# ============================================================================
# BANDIT ALGORITHMS
# ============================================================================

class LambdaBandit:
    """
    Multi-armed bandit with Lambda exploration bonus.

    Key innovation: Exploration bonus = (1 - exp(-Lambda²))
    """

    def __init__(self, n_arms, c=2.0):
        """
        Initialize Lambda-Bandit.

        Args:
            n_arms: Number of arms
            c: Exploration constant
        """
        self.n_arms = n_arms
        self.c = c

        # Statistics for each arm
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.sum_rewards = np.zeros(n_arms)
        self.sum_squared_rewards = np.zeros(n_arms)

    def select_arm(self):
        """Select arm using UCB + Lambda bonus"""
        # Initialize: pull each arm once
        if np.any(self.counts == 0):
            return np.argmax(self.counts == 0)

        # Compute UCB values
        ucb_values = self.values + self.lambda_exploration_bonus()

        return np.argmax(ucb_values)

    def lambda_exploration_bonus(self):
        """
        Compute Lambda-based exploration bonus.

        For each arm:
        1. Calculate Lambda between arm and global average
        2. Compute uncertainty = 1 - exp(-Lambda²)
        3. Scale by UCB term: c * sqrt(log(t)/n)
        """
        bonuses = []
        t = np.sum(self.counts)

        # Global average distribution
        global_mu = np.mean(self.values)
        global_sigma = np.std(self.values) + 1e-8

        for arm in range(self.n_arms):
            # Get arm's distribution estimate
            arm_mu = self.values[arm]
            arm_sigma = self.estimate_sigma(arm)

            # Calculate Lambda
            dist_arm = {'mu': arm_mu, 'sigma': arm_sigma}
            dist_global = {'mu': global_mu, 'sigma': global_sigma}

            Lambda = compute_distribution_lambda(dist_arm, dist_global)

            # Uncertainty from Lambda
            confidence = np.exp(-Lambda**2)
            uncertainty = 1 - confidence

            # UCB scaling term
            n = max(1, self.counts[arm])
            ucb_term = np.sqrt(np.log(t + 1) / n)

            # Combine
            bonus = self.c * uncertainty * ucb_term

            bonuses.append(bonus)

        return np.array(bonuses)

    def estimate_sigma(self, arm):
        """Estimate standard deviation for arm"""
        n = max(1, self.counts[arm])

        # Sample variance
        mean_sq = self.sum_squared_rewards[arm] / n
        mean = self.values[arm]
        variance = max(0, mean_sq - mean**2)

        sigma = np.sqrt(variance) + 1e-8  # Add small constant for stability

        return sigma

    def update(self, arm, reward):
        """Update statistics after observing reward"""
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.sum_squared_rewards[arm] += reward**2

        # Update mean
        self.values[arm] = self.sum_rewards[arm] / self.counts[arm]


class UCB1:
    """Standard UCB1 algorithm"""

    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        # Initialize
        if np.any(self.counts == 0):
            return np.argmax(self.counts == 0)

        # UCB1 formula
        t = np.sum(self.counts)
        ucb_values = self.values + self.c * np.sqrt(np.log(t + 1) / (self.counts + 1e-8))

        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]

        # Running average
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward


class EpsilonGreedy:
    """Epsilon-greedy algorithm"""

    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward


class ThompsonSampling:
    """Thompson Sampling for Gaussian bandits"""

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.mu_estimates = np.zeros(n_arms)
        self.sigma_estimates = np.ones(n_arms)

        # Priors (normal-gamma conjugate)
        self.alpha = np.ones(n_arms)  # Shape
        self.beta = np.ones(n_arms)   # Rate

    def select_arm(self):
        # Sample from posterior
        samples = []
        for arm in range(self.n_arms):
            # Sample precision from Gamma
            precision = np.random.gamma(self.alpha[arm], 1.0 / self.beta[arm])
            sigma = 1.0 / np.sqrt(precision + 1e-8)

            # Sample mean from Normal
            mu = np.random.normal(self.mu_estimates[arm], sigma / np.sqrt(self.counts[arm] + 1))

            samples.append(mu)

        return np.argmax(samples)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]

        # Update mean estimate
        old_mu = self.mu_estimates[arm]
        self.mu_estimates[arm] = ((n - 1) / n) * old_mu + (1 / n) * reward

        # Update variance estimate (simplified)
        self.alpha[arm] += 0.5
        self.beta[arm] += 0.5 * (reward - old_mu)**2 * (n - 1) / n


# ============================================================================
# SIMULATION
# ============================================================================

def sample_reward(door_idx):
    """Sample reward from door"""
    door = DOORS[door_idx]
    return np.random.normal(door['mu'], door['sigma'])


def run_bandit(BanditClass, horizon=500, **kwargs):
    """
    Run single bandit trial.

    Returns:
        Dictionary with performance metrics
    """
    bandit = BanditClass(n_arms=len(DOORS), **kwargs)

    # Track history
    arm_history = []
    reward_history = []
    regret_history = []

    optimal_value = DOORS[OPTIMAL_ARM]['mu']
    cumulative_regret = 0

    for t in range(horizon):
        # Select arm
        arm = bandit.select_arm()
        arm_history.append(arm)

        # Sample reward
        reward = sample_reward(arm)
        reward_history.append(reward)

        # Update bandit
        bandit.update(arm, reward)

        # Track regret
        regret = optimal_value - DOORS[arm]['mu']
        cumulative_regret += regret
        regret_history.append(cumulative_regret)

    # Get final values (different bandits use different attribute names)
    if hasattr(bandit, 'values'):
        final_values = bandit.values
    elif hasattr(bandit, 'mu_estimates'):
        final_values = bandit.mu_estimates
    else:
        final_values = np.zeros(bandit.n_arms)

    return {
        'arm_history': arm_history,
        'reward_history': reward_history,
        'regret_history': regret_history,
        'final_counts': bandit.counts,
        'final_values': final_values
    }


def convergence_time(arm_history, window=10):
    """
    Calculate time to convergence.

    Converged = last `window` selections are all optimal arm
    """
    for t in range(len(arm_history) - window):
        if all(arm == OPTIMAL_ARM for arm in arm_history[t:t+window]):
            return t + window

    return len(arm_history)  # Never converged


def test_reddit_problem(n_trials=1000, horizon=500):
    """
    Test all algorithms on Reddit 3-door problem.

    Args:
        n_trials: Number of independent trials
        horizon: Steps per trial

    Returns:
        Results dictionary
    """
    print("="*80)
    print("REDDIT 3-DOOR PROBLEM")
    print("="*80)
    print()

    print("Ground Truth:")
    print("-"*40)
    for i, door in enumerate(DOORS):
        optimal_mark = " [OPTIMAL]" if i == OPTIMAL_ARM else ""
        print(f"  {door['name']}: N(mu={door['mu']}, sigma={door['sigma']}){optimal_mark}")
    print()

    print(f"Running {n_trials} trials, {horizon} steps each...")
    print()

    # Algorithms to test
    algorithms = {
        'Lambda-Bandit': (LambdaBandit, {}),
        'UCB1': (UCB1, {}),
        'Thompson Sampling': (ThompsonSampling, {}),
        'Epsilon-Greedy': (EpsilonGreedy, {'epsilon': 0.1})
    }

    results = {}

    for name, (BanditClass, kwargs) in algorithms.items():
        print(f"Testing {name}...")

        convergence_times = []
        final_regrets = []
        optimal_selections = []

        for trial in range(n_trials):
            result = run_bandit(BanditClass, horizon=horizon, **kwargs)

            # Metrics
            conv_time = convergence_time(result['arm_history'])
            convergence_times.append(conv_time)

            final_regrets.append(result['regret_history'][-1])

            # % optimal arm in final 100 steps
            final_100 = result['arm_history'][-100:]
            optimal_pct = 100 * sum(1 for a in final_100 if a == OPTIMAL_ARM) / len(final_100)
            optimal_selections.append(optimal_pct)

        results[name] = {
            'convergence_time': np.mean(convergence_times),
            'convergence_std': np.std(convergence_times),
            'final_regret': np.mean(final_regrets),
            'regret_std': np.std(final_regrets),
            'optimal_pct': np.mean(optimal_selections),
            'optimal_std': np.std(optimal_selections)
        }

    return results


def print_results(results):
    """Print comparison table"""
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()

    # Find best convergence time
    best_conv_time = min(r['convergence_time'] for r in results.values())

    print("Convergence Time (steps to 10 consecutive optimal selections):")
    print("-"*80)
    print(f"{'Algorithm':<20s} {'Mean':>10s} {'Std':>10s} {'vs Best':>15s}")
    print("-"*80)

    for name, data in results.items():
        pct_diff = 100 * (data['convergence_time'] / best_conv_time - 1)
        marker = " [BEST]" if data['convergence_time'] == best_conv_time else ""

        print(f"{name:<20s} {data['convergence_time']:>10.1f} {data['convergence_std']:>10.1f} "
              f"{pct_diff:>+14.1f}%{marker}")
    print()

    print("Cumulative Regret (lower is better):")
    print("-"*80)
    print(f"{'Algorithm':<20s} {'Mean':>10s} {'Std':>10s}")
    print("-"*80)

    for name, data in results.items():
        print(f"{name:<20s} {data['final_regret']:>10.1f} {data['regret_std']:>10.1f}")
    print()

    print("% Optimal Arm in Final 100 Steps (higher is better):")
    print("-"*80)
    print(f"{'Algorithm':<20s} {'Mean':>10s} {'Std':>10s}")
    print("-"*80)

    for name, data in results.items():
        print(f"{name:<20s} {data['optimal_pct']:>10.1f}% {data['optimal_std']:>10.1f}%")
    print()


def plot_results(results, n_trials=100, horizon=500):
    """Generate comparison plots"""
    print("Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Run a few trials for each algorithm to plot
    algorithms = {
        'Lambda-Bandit': (LambdaBandit, {}),
        'UCB1': (UCB1, {}),
        'Thompson Sampling': (ThompsonSampling, {}),
        'Epsilon-Greedy': (EpsilonGreedy, {'epsilon': 0.1})
    }

    colors = {'Lambda-Bandit': 'red', 'UCB1': 'blue',
              'Thompson Sampling': 'green', 'Epsilon-Greedy': 'orange'}

    # Plot 1: Regret curves
    for name, (BanditClass, kwargs) in algorithms.items():
        all_regrets = []

        for _ in range(n_trials):
            result = run_bandit(BanditClass, horizon=horizon, **kwargs)
            all_regrets.append(result['regret_history'])

        mean_regret = np.mean(all_regrets, axis=0)
        std_regret = np.std(all_regrets, axis=0)

        t = np.arange(horizon)
        axes[0, 0].plot(t, mean_regret, label=name, color=colors[name], linewidth=2)
        axes[0, 0].fill_between(t, mean_regret - std_regret, mean_regret + std_regret,
                                alpha=0.2, color=colors[name])

    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Cumulative Regret')
    axes[0, 0].set_title('Regret Curves (Mean ± Std)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Arm selection over time
    for name, (BanditClass, kwargs) in algorithms.items():
        # Single representative trial
        result = run_bandit(BanditClass, horizon=horizon, **kwargs)

        # Rolling average: % optimal arm in last 50 steps
        window = 50
        optimal_pct = []
        for i in range(window, len(result['arm_history'])):
            recent = result['arm_history'][i-window:i]
            pct = 100 * sum(1 for a in recent if a == OPTIMAL_ARM) / window
            optimal_pct.append(pct)

        t = np.arange(window, horizon)
        axes[0, 1].plot(t, optimal_pct, label=name, color=colors[name], linewidth=2)

    axes[0, 1].axhline(100, color='black', linestyle='--', alpha=0.5, label='Perfect')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('% Optimal Arm (Rolling 50)')
    axes[0, 1].set_title('Convergence to Optimal Arm')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Convergence time comparison
    conv_times = [results[name]['convergence_time'] for name in algorithms.keys()]
    conv_stds = [results[name]['convergence_std'] for name in algorithms.keys()]

    x_pos = np.arange(len(algorithms))
    bars = axes[1, 0].bar(x_pos, conv_times, yerr=conv_stds, capsize=5,
                          color=[colors[name] for name in algorithms.keys()])

    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(algorithms.keys(), rotation=45, ha='right')
    axes[1, 0].set_ylabel('Steps to Convergence')
    axes[1, 0].set_title('Convergence Speed')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Final regret comparison
    final_regrets = [results[name]['final_regret'] for name in algorithms.keys()]
    regret_stds = [results[name]['regret_std'] for name in algorithms.keys()]

    bars = axes[1, 1].bar(x_pos, final_regrets, yerr=regret_stds, capsize=5,
                          color=[colors[name] for name in algorithms.keys()])

    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(algorithms.keys(), rotation=45, ha='right')
    axes[1, 1].set_ylabel('Cumulative Regret')
    axes[1, 1].set_title('Final Regret')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('reddit_3door_comparison.png', dpi=150)
    print("Saved: reddit_3door_comparison.png")
    print()


def main():
    """Run Reddit 3-door problem test"""
    print("\n")
    print("="*80)
    print("REDDIT 3-DOOR PROBLEM - LAMBDA-BANDIT VALIDATION")
    print("="*80)
    print()

    print("Hypothesis: Lambda-Bandit converges faster than baselines")
    print("Challenge: High variance in Door 2 makes it hard to identify")
    print()

    # Run tests
    results = test_reddit_problem(n_trials=1000, horizon=500)

    # Print results
    print_results(results)

    # Plot
    plot_results(results, n_trials=100, horizon=500)

    # Patent validation check
    print("="*80)
    print("PATENT CLAIM VALIDATION")
    print("="*80)
    print()

    lambda_conv = results['Lambda-Bandit']['convergence_time']
    ucb1_conv = results['UCB1']['convergence_time']
    improvement = 100 * (1 - lambda_conv / ucb1_conv)

    print(f"Lambda-Bandit convergence: {lambda_conv:.1f} steps")
    print(f"UCB1 convergence: {ucb1_conv:.1f} steps")
    print(f"Improvement: {improvement:.1f}%")
    print()

    if improvement > 0:
        print("[SUCCESS] Lambda-Bandit faster than UCB1")
    else:
        print("[NEEDS WORK] Lambda-Bandit not faster - tune hyperparameters")

    print()

    if improvement > 30:
        print("[PATENT STRONG] >30% improvement achieved!")
    elif improvement > 10:
        print("[PATENT VIABLE] >10% improvement achieved")
    else:
        print("[PATENT WEAK] <10% improvement - more validation needed")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
