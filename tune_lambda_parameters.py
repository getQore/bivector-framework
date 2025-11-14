#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lambda-Bandit Hyperparameter Tuning
====================================

Test different parameter configurations to find where Lambda-Bandit excels.

Goal: Find optimal c and exploration formula for Reddit 3-door problem
"""

import numpy as np
import matplotlib.pyplot as plt
from distribution_bivector_utils import gaussian_to_bivector, compute_distribution_lambda

# Reddit 3-door problem (same as Day 2)
DOORS = [
    {'name': 'Door 1', 'mu': 1.0, 'sigma': 0.1},
    {'name': 'Door 2', 'mu': 2.0, 'sigma': 2.0},  # Optimal
    {'name': 'Door 3', 'mu': 1.5, 'sigma': 0.5}
]
OPTIMAL_ARM = 1


class LambdaBanditTunable:
    """Lambda-Bandit with tunable parameters"""

    def __init__(self, n_arms, c=2.0, formula='standard'):
        """
        Args:
            c: Scaling factor for exploration bonus
            formula: One of:
                - 'standard': (1 - exp(-Λ²)) * c * ucb_term
                - 'inverse': exp(-c*Λ²) * ucb_term  (suppress when Λ high)
                - 'linear': c * Λ * ucb_term
                - 'bounded': Λ/(1+Λ) * c * ucb_term
                - 'first_order': (1 - exp(-c*Λ)) * ucb_term
        """
        self.n_arms = n_arms
        self.c = c
        self.formula = formula

        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.stds = np.ones(n_arms)

    def compute_arm_lambda(self, arm):
        """Compute Λ between arm and global average"""
        # Global average
        if np.sum(self.counts) == 0:
            return 0.0

        global_mu = np.average(self.values, weights=self.counts + 1e-8)
        global_sigma = np.sqrt(np.average((self.values - global_mu)**2,
                                          weights=self.counts + 1e-8))

        # Arm distribution
        arm_dist = {'mu': self.values[arm], 'sigma': self.stds[arm]}
        global_dist = {'mu': global_mu, 'sigma': global_sigma}

        Lambda = compute_distribution_lambda(arm_dist, global_dist)
        return Lambda

    def select_arm(self):
        """UCB with Lambda exploration bonus"""
        # Initialize: pull each arm once
        if np.any(self.counts == 0):
            return np.argmax(self.counts == 0)

        bonuses = []
        t = np.sum(self.counts)

        for arm in range(self.n_arms):
            # Calculate Λ
            Lambda = self.compute_arm_lambda(arm)

            # UCB term
            n = max(1, self.counts[arm])
            ucb_term = np.sqrt(np.log(t + 1) / n)

            # Exploration bonus based on formula
            if self.formula == 'standard':
                # Original: (1 - exp(-Λ²)) * c * ucb
                uncertainty = 1 - np.exp(-Lambda**2)
                bonus = uncertainty * self.c * ucb_term

            elif self.formula == 'inverse':
                # Inverted: exp(-c*Λ²) * ucb (suppress when Λ high)
                bonus = np.exp(-self.c * Lambda**2) * ucb_term

            elif self.formula == 'linear':
                # Linear: c * Λ * ucb
                bonus = self.c * Lambda * ucb_term

            elif self.formula == 'bounded':
                # Bounded: Λ/(1+Λ) * c * ucb
                bonus = (Lambda / (1 + Lambda)) * self.c * ucb_term

            elif self.formula == 'first_order':
                # First-order: (1 - exp(-c*Λ)) * ucb
                uncertainty = 1 - np.exp(-self.c * Lambda)
                bonus = uncertainty * ucb_term

            else:
                raise ValueError(f"Unknown formula: {self.formula}")

            bonuses.append(bonus)

        ucb_values = self.values + np.array(bonuses)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """Update arm statistics"""
        self.counts[arm] += 1
        n = self.counts[arm]

        # Running mean
        old_value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * old_value + (1 / n) * reward

        # Running std (Welford's algorithm - simplified)
        old_std = self.stds[arm]
        delta = reward - old_value
        self.stds[arm] = np.sqrt(((n - 1) * old_std**2 + delta * (reward - self.values[arm])) / n + 1e-8)


def sample_reward(door_idx):
    """Sample reward from door"""
    door = DOORS[door_idx]
    return np.random.normal(door['mu'], door['sigma'])


def run_bandit(c, formula, n_trials=100, horizon=500):
    """
    Run bandit with given parameters.

    Args:
        c: Scaling factor
        formula: Exploration formula
        n_trials: Number of trials (reduced from 1000 for speed)
        horizon: Steps per trial

    Returns:
        Dictionary with metrics
    """
    convergence_times = []
    final_regrets = []

    for trial in range(n_trials):
        bandit = LambdaBanditTunable(n_arms=len(DOORS), c=c, formula=formula)

        regret = 0
        convergence_time = horizon
        recent_arms = []

        for t in range(horizon):
            # Select and pull arm
            arm = bandit.select_arm()
            reward = sample_reward(arm)
            bandit.update(arm, reward)

            # Track regret
            optimal_value = DOORS[OPTIMAL_ARM]['mu']
            regret += optimal_value - DOORS[arm]['mu']

            # Check convergence (10 consecutive optimal)
            recent_arms.append(arm)
            if len(recent_arms) > 10:
                recent_arms.pop(0)

            if len(recent_arms) == 10 and all(a == OPTIMAL_ARM for a in recent_arms):
                if convergence_time == horizon:
                    convergence_time = t

        convergence_times.append(convergence_time)
        final_regrets.append(regret)

    return {
        'convergence_time': np.mean(convergence_times),
        'convergence_std': np.std(convergence_times),
        'final_regret': np.mean(final_regrets),
        'regret_std': np.std(final_regrets)
    }


def test_parameter_sweep():
    """
    Comprehensive parameter sweep.

    Test all combinations of c and formula.
    """
    print("="*80)
    print("LAMBDA-BANDIT PARAMETER TUNING")
    print("="*80)
    print()
    print("Testing on Reddit 3-door problem with 100 trials (faster)")
    print()

    # Parameter grid
    c_values = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    formulas = ['standard', 'inverse', 'linear', 'bounded', 'first_order']

    results = {}

    # Baseline (UCB1 equivalent: c=2.0, inverse formula gives pure UCB)
    print("Running baseline (UCB1-like)...")
    baseline = run_bandit(c=2.0, formula='inverse', n_trials=100)
    print(f"Baseline convergence: {baseline['convergence_time']:.1f} steps")
    print()

    # Test all combinations
    total_tests = len(c_values) * len(formulas)
    test_num = 0

    for formula in formulas:
        for c in c_values:
            test_num += 1
            print(f"[{test_num}/{total_tests}] Testing formula={formula}, c={c:.1f}...", end=" ")

            result = run_bandit(c=c, formula=formula, n_trials=100)
            results[(c, formula)] = result

            improvement = 100 * (1 - result['convergence_time'] / baseline['convergence_time'])
            print(f"Conv={result['convergence_time']:.1f} ({improvement:+.1f}% vs baseline)")

    return results, baseline


def analyze_results(results, baseline):
    """Find best parameters"""
    print()
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()

    # Sort by convergence time
    sorted_results = sorted(results.items(),
                           key=lambda x: x[1]['convergence_time'])

    print("Top 10 configurations:")
    print("-"*80)
    print(f"{'Rank':<6} {'Formula':<15} {'c':<6} {'Conv Time':<12} {'vs Baseline':<12} {'Regret'}")
    print("-"*80)

    baseline_conv = baseline['convergence_time']

    for i, ((c, formula), result) in enumerate(sorted_results[:10]):
        improvement = 100 * (1 - result['convergence_time'] / baseline_conv)
        print(f"{i+1:<6} {formula:<15} {c:<6.1f} {result['convergence_time']:<12.1f} "
              f"{improvement:+11.1f}% {result['final_regret']:8.1f}")

    print()

    # Best configuration
    best_config, best_result = sorted_results[0]
    best_c, best_formula = best_config
    improvement = 100 * (1 - best_result['convergence_time'] / baseline_conv)

    print("="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"Formula: {best_formula}")
    print(f"c: {best_c:.1f}")
    print(f"Convergence: {best_result['convergence_time']:.1f} steps")
    print(f"Improvement: {improvement:+.1f}% vs baseline")
    print(f"Regret: {best_result['final_regret']:.1f}")
    print()

    if improvement > 10:
        print("[SUCCESS] >10% improvement found!")
        print("Lambda-Bandit is viable with tuned parameters.")
    elif improvement > 0:
        print("[MARGINAL] Small improvement found.")
        print("May not be significant enough for patent claim.")
    else:
        print("[NEGATIVE] No improvement over baseline.")
        print("Lambda-Bandit struggles on Reddit problem even with tuning.")

    print()

    return best_config, best_result


def plot_results(results, baseline):
    """Visualize parameter space"""
    # Extract data
    c_values = sorted(set(c for c, _ in results.keys()))
    formulas = sorted(set(f for _, f in results.keys()))

    # Create heatmap data
    convergence_grid = np.zeros((len(formulas), len(c_values)))

    for i, formula in enumerate(formulas):
        for j, c in enumerate(c_values):
            conv_time = results[(c, formula)]['convergence_time']
            # Improvement over baseline
            improvement = 100 * (1 - conv_time / baseline['convergence_time'])
            convergence_grid[i, j] = improvement

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(convergence_grid, cmap='RdYlGn', aspect='auto',
                   vmin=-50, vmax=50, origin='lower')

    # Labels
    ax.set_xticks(range(len(c_values)))
    ax.set_xticklabels([f"{c:.1f}" for c in c_values])
    ax.set_xlabel('Scaling Factor (c)', fontsize=12)

    ax.set_yticks(range(len(formulas)))
    ax.set_yticklabels(formulas)
    ax.set_ylabel('Exploration Formula', fontsize=12)

    ax.set_title('Lambda-Bandit Parameter Tuning\n(% Improvement over Baseline)',
                 fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% Improvement', rotation=270, labelpad=20)

    # Annotations
    for i in range(len(formulas)):
        for j in range(len(c_values)):
            text = ax.text(j, i, f"{convergence_grid[i, j]:.0f}%",
                          ha="center", va="center", color="black", fontsize=9)

    # Mark baseline (0% improvement)
    ax.axhline(y=-0.5, color='blue', linestyle='--', linewidth=2, alpha=0.5,
               label='Baseline (0%)')

    plt.tight_layout()
    plt.savefig('lambda_parameter_tuning.png', dpi=150)
    print("Saved: lambda_parameter_tuning.png")
    print()


def main():
    """Run parameter tuning"""
    print("\n")

    # Run sweep
    results, baseline = test_parameter_sweep()

    # Analyze
    best_config, best_result = analyze_results(results, baseline)

    # Plot
    plot_results(results, baseline)

    # Save results
    import json

    output = {
        'baseline': baseline,
        'best_config': {
            'c': best_config[0],
            'formula': best_config[1],
            'convergence_time': best_result['convergence_time'],
            'final_regret': best_result['final_regret']
        },
        'all_results': {
            f"c={c:.1f}_formula={formula}": result
            for (c, formula), result in results.items()
        }
    }

    with open('lambda_tuning_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("Saved: lambda_tuning_results.json")
    print()

    print("="*80)
    print("PARAMETER TUNING COMPLETE")
    print("="*80)
    print()

    return best_config, best_result


if __name__ == "__main__":
    best_config, best_result = main()
