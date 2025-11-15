#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lambda-Bandit Niche Testing
============================

Test optimized Lambda-Bandit on problems where it should excel:
1. Reddit problem (with optimized parameters vs real baselines)
2. Many-armed bandits (K = 20, 50)
3. Correlated arms
4. Non-stationary rewards

Based on tuning results: Use bounded formula with c=5.0
"""

import numpy as np
import matplotlib.pyplot as plt
from distribution_bivector_utils import gaussian_to_bivector, compute_distribution_lambda

# Import baseline bandit algorithms from reddit_3door_problem
import sys
sys.path.insert(0, '.')

# Reddit 3-door problem
REDDIT_DOORS = [
    {'name': 'Door 1', 'mu': 1.0, 'sigma': 0.1},
    {'name': 'Door 2', 'mu': 2.0, 'sigma': 2.0},  # Optimal
    {'name': 'Door 3', 'mu': 1.5, 'sigma': 0.5}
]


class LambdaBanditOptimized:
    """Lambda-Bandit with optimized parameters (bounded formula, c=5.0)"""

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.stds = np.ones(n_arms)
        self.c = 5.0  # Optimized!
        self.formula = 'bounded'  # Optimized!

    def compute_arm_lambda(self, arm):
        """Compute Λ between arm and global average"""
        if np.sum(self.counts) == 0:
            return 0.0

        global_mu = np.average(self.values, weights=self.counts + 1e-8)
        global_sigma = np.sqrt(np.average((self.values - global_mu)**2,
                                          weights=self.counts + 1e-8))

        arm_dist = {'mu': self.values[arm], 'sigma': self.stds[arm]}
        global_dist = {'mu': global_mu, 'sigma': global_sigma}

        Lambda = compute_distribution_lambda(arm_dist, global_dist)
        return Lambda

    def select_arm(self):
        """UCB with Lambda exploration bonus"""
        if np.any(self.counts == 0):
            return np.argmax(self.counts == 0)

        bonuses = []
        t = np.sum(self.counts)

        for arm in range(self.n_arms):
            Lambda = self.compute_arm_lambda(arm)
            n = max(1, self.counts[arm])
            ucb_term = np.sqrt(np.log(t + 1) / n)

            # Bounded formula: Λ/(1+Λ) * c * ucb
            bonus = (Lambda / (1 + Lambda)) * self.c * ucb_term
            bonuses.append(bonus)

        ucb_values = self.values + np.array(bonuses)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """Update arm statistics"""
        self.counts[arm] += 1
        n = self.counts[arm]

        old_value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * old_value + (1 / n) * reward

        old_std = self.stds[arm]
        delta = reward - old_value
        self.stds[arm] = np.sqrt(((n - 1) * old_std**2 + delta * (reward - self.values[arm])) / n + 1e-8)


class UCB1:
    """Standard UCB1"""
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.c = 2.0

    def select_arm(self):
        if np.any(self.counts == 0):
            return np.argmax(self.counts == 0)

        t = np.sum(self.counts)
        ucb_values = self.values + self.c * np.sqrt(np.log(t + 1) / (self.counts + 1e-8))
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward


class ThompsonSampling:
    """Thompson Sampling"""
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.mu_estimates = np.zeros(n_arms)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        samples = []
        for arm in range(self.n_arms):
            precision = np.random.gamma(self.alpha[arm], 1.0 / self.beta[arm])
            sigma = 1.0 / np.sqrt(precision + 1e-8)
            mu = np.random.normal(self.mu_estimates[arm], sigma / np.sqrt(self.counts[arm] + 1))
            samples.append(mu)
        return np.argmax(samples)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        old_mu = self.mu_estimates[arm]
        self.mu_estimates[arm] = ((n - 1) / n) * old_mu + (1 / n) * reward
        self.alpha[arm] += 0.5
        self.beta[arm] += 0.5 * (reward - old_mu)**2 * (n - 1) / n


def run_bandit(BanditClass, doors, optimal_arm, n_trials=100, horizon=500):
    """Run bandit algorithm"""
    convergence_times = []
    final_regrets = []

    for trial in range(n_trials):
        bandit = BanditClass(n_arms=len(doors))
        regret = 0
        convergence_time = horizon
        recent_arms = []

        for t in range(horizon):
            arm = bandit.select_arm()
            reward = np.random.normal(doors[arm]['mu'], doors[arm]['sigma'])
            bandit.update(arm, reward)

            optimal_value = doors[optimal_arm]['mu']
            regret += optimal_value - doors[arm]['mu']

            recent_arms.append(arm)
            if len(recent_arms) > 10:
                recent_arms.pop(0)

            if len(recent_arms) == 10 and all(a == optimal_arm for a in recent_arms):
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


def test_reddit_optimized():
    """Test optimized Lambda-Bandit on Reddit problem vs real baselines"""
    print("="*80)
    print("TEST 1: REDDIT 3-DOOR PROBLEM (Optimized Lambda)")
    print("="*80)
    print()

    algorithms = {
        'Lambda-Bandit (Optimized)': LambdaBanditOptimized,
        'UCB1': UCB1,
        'Thompson Sampling': ThompsonSampling
    }

    results = {}
    for name, BanditClass in algorithms.items():
        print(f"Running {name}... (100 trials)", end=" ")
        result = run_bandit(BanditClass, REDDIT_DOORS, optimal_arm=1, n_trials=100)
        results[name] = result
        print(f"Conv={result['convergence_time']:.1f} steps")

    print()
    print("Results:")
    print("-"*80)
    print(f"{'Algorithm':<30} {'Convergence':<15} {'vs Best':<12} {'Regret'}")
    print("-"*80)

    best_conv = min(r['convergence_time'] for r in results.values())

    for name, result in results.items():
        improvement = 100 * (result['convergence_time'] / best_conv - 1)
        marker = " [BEST]" if result['convergence_time'] == best_conv else ""
        print(f"{name:<30} {result['convergence_time']:<15.1f} {improvement:+11.1f}%{marker:>12} {result['final_regret']:8.1f}")

    print()
    return results


def test_many_arms():
    """Test on many-armed bandits (K=20, K=50)"""
    print("="*80)
    print("TEST 2: MANY-ARMED BANDITS")
    print("="*80)
    print()

    results_by_k = {}

    for K in [20, 50]:
        print(f"\nK = {K} arms:")
        print("-"*80)

        # Generate random problem
        np.random.seed(42 + K)  # Reproducible
        doors = [
            {'name': f'Arm {i}', 'mu': np.random.uniform(0, 10), 'sigma': np.random.uniform(0.5, 2.0)}
            for i in range(K)
        ]
        optimal_arm = np.argmax([d['mu'] for d in doors])

        algorithms = {
            'Lambda-Bandit': LambdaBanditOptimized,
            'UCB1': UCB1,
            'Thompson': ThompsonSampling
        }

        results = {}
        for name, BanditClass in algorithms.items():
            print(f"  {name}...", end=" ")
            result = run_bandit(BanditClass, doors, optimal_arm, n_trials=50, horizon=1000)
            results[name] = result
            print(f"Conv={result['convergence_time']:.1f}")

        results_by_k[K] = results

    # Analysis
    print()
    print("Scaling Analysis:")
    print("-"*80)
    print(f"{'K':<6} {'Algorithm':<20} {'Convergence':<15} {'vs Lambda'}")
    print("-"*80)

    for K, results in results_by_k.items():
        lambda_conv = results['Lambda-Bandit']['convergence_time']
        for name, result in results.items():
            if name == 'Lambda-Bandit':
                marker = " [REFERENCE]"
                pct = 0.0
            else:
                pct = 100 * (result['convergence_time'] / lambda_conv - 1)
                marker = ""
            print(f"{K:<6} {name:<20} {result['convergence_time']:<15.1f} {pct:+11.1f}%{marker}")

    print()
    return results_by_k


def test_correlated_arms():
    """Test on correlated arms problem"""
    print("="*80)
    print("TEST 3: CORRELATED ARMS")
    print("="*80)
    print()
    print("Problem: 10 arms with pairwise correlations")
    print("Lambda should capture correlation structure geometrically")
    print()

    # Generate correlated arms
    np.random.seed(123)
    n_arms = 10

    # Base means
    base_means = np.random.uniform(0, 10, n_arms)

    # Create correlation structure
    correlation_groups = [[0, 1, 2], [3, 4], [5, 6, 7], [8, 9]]

    doors = []
    for i in range(n_arms):
        # Find group
        for group in correlation_groups:
            if i in group:
                # Correlated within group
                group_mean = np.mean([base_means[j] for j in group])
                mu = group_mean + np.random.normal(0, 0.5)
                break
        else:
            mu = base_means[i]

        doors.append({'name': f'Arm {i}', 'mu': mu, 'sigma': 1.0})

    optimal_arm = np.argmax([d['mu'] for d in doors])

    algorithms = {
        'Lambda-Bandit': LambdaBanditOptimized,
        'UCB1': UCB1,
        'Thompson': ThompsonSampling
    }

    results = {}
    for name, BanditClass in algorithms.items():
        print(f"{name}...", end=" ")
        result = run_bandit(BanditClass, doors, optimal_arm, n_trials=50, horizon=500)
        results[name] = result
        print(f"Conv={result['convergence_time']:.1f}")

    print()
    print("Results:")
    print("-"*80)
    lambda_conv = results['Lambda-Bandit']['convergence_time']
    for name, result in results.items():
        if name == 'Lambda-Bandit':
            pct_str = ""
        else:
            pct = 100 * (result['convergence_time'] / lambda_conv - 1)
            pct_str = f" ({pct:+.1f}% vs Lambda)"
        print(f"{name:<20}: {result['convergence_time']:.1f} steps{pct_str}")

    print()
    return results


def main():
    """Run all niche tests"""
    print("\n")
    print("="*80)
    print("LAMBDA-BANDIT NICHE TESTING")
    print("="*80)
    print()
    print("Using optimized parameters:")
    print("  Formula: bounded (Λ/(1+Λ))")
    print("  Scaling: c = 5.0")
    print()

    # Test 1: Reddit with optimized parameters
    reddit_results = test_reddit_optimized()

    # Test 2: Many arms
    many_arms_results = test_many_arms()

    # Test 3: Correlated arms
    correlated_results = test_correlated_arms()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()

    # Reddit
    lambda_reddit = reddit_results['Lambda-Bandit (Optimized)']['convergence_time']
    ucb1_reddit = reddit_results['UCB1']['convergence_time']
    thompson_reddit = reddit_results['Thompson Sampling']['convergence_time']

    improvement_vs_ucb1 = 100 * (1 - lambda_reddit / ucb1_reddit)
    improvement_vs_thompson = 100 * (1 - lambda_reddit / thompson_reddit)

    print(f"Reddit 3-Door:")
    print(f"  Lambda: {lambda_reddit:.1f} steps")
    print(f"  UCB1: {ucb1_reddit:.1f} steps ({improvement_vs_ucb1:+.1f}%)")
    print(f"  Thompson: {thompson_reddit:.1f} steps ({improvement_vs_thompson:+.1f}%)")
    print()

    if improvement_vs_ucb1 > 10 or improvement_vs_thompson > 10:
        print("[SUCCESS] Lambda-Bandit >10% better on Reddit problem!")
    elif improvement_vs_ucb1 > 0 or improvement_vs_thompson > 0:
        print("[MARGINAL] Lambda-Bandit slightly better")
    else:
        print("[NEGATIVE] Lambda-Bandit still slower")

    print()

    # Many arms
    print("Many Arms:")
    for K, results in many_arms_results.items():
        lambda_conv = results['Lambda-Bandit']['convergence_time']
        ucb1_conv = results['UCB1']['convergence_time']
        improvement = 100 * (1 - lambda_conv / ucb1_conv)
        print(f"  K={K}: Lambda {lambda_conv:.1f}, UCB1 {ucb1_conv:.1f} ({improvement:+.1f}%)")

    print()
    print("="*80)
    print()


if __name__ == "__main__":
    main()
