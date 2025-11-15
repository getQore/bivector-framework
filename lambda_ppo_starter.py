#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lambda-PPO: PPO with Bivector-Based Confidence Weighting
=========================================================

Extends Proximal Policy Optimization (PPO) with Lambda weighting:
- Calculate Λ = ||[B_old_policy, B_new_policy]||
- Weight advantages by confidence = exp(-Λ²)
- Add exploration bonus = 1 - exp(-Λ²)

This is a STARTER implementation for patent validation.

Rick Mathews - RL Patent Application
November 14, 2024
"""

import numpy as np
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. Lambda-PPO requires PyTorch.")
    print("Install with: pip install torch")

from distribution_bivector_utils import BivectorCl31, compute_distribution_lambda


# ============================================================================
# NEURAL NETWORKS
# ============================================================================

if TORCH_AVAILABLE:
    class PolicyNetwork(nn.Module):
        """
        Policy network: state -> action probabilities
        """

        def __init__(self, state_dim, action_dim, hidden_dim=64):
            super(PolicyNetwork, self).__init__()

            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)

        def forward(self, state):
            """
            Forward pass.

            Args:
                state: Tensor of shape (batch, state_dim)

            Returns:
                logits: Tensor of shape (batch, action_dim)
            """
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            logits = self.fc3(x)
            return logits

        def get_distribution(self, state):
            """Get action distribution"""
            logits = self.forward(state)
            return Categorical(logits=logits)

        def log_prob(self, state, action):
            """Get log probability of action"""
            dist = self.get_distribution(state)
            return dist.log_prob(action)


    class ValueNetwork(nn.Module):
        """
        Value network: state -> value estimate
        """

        def __init__(self, state_dim, hidden_dim=64):
            super(ValueNetwork, self).__init__()

            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

        def forward(self, state):
            """
            Forward pass.

            Args:
                state: Tensor of shape (batch, state_dim)

            Returns:
                value: Tensor of shape (batch, 1)
            """
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            value = self.fc3(x)
            return value.squeeze(-1)


# ============================================================================
# LAMBDA-PPO AGENT
# ============================================================================

if TORCH_AVAILABLE:
    class LambdaPPO:
        """
        Proximal Policy Optimization with Lambda weighting.

        Key modifications to standard PPO:
        1. Compute Λ between old and new policy distributions
        2. Weight advantages by confidence = exp(-Λ²)
        3. Add exploration bonus = 1 - exp(-Λ²)
        """

        def __init__(self, state_dim, action_dim, hidden_dim=64,
                     lr_policy=3e-4, lr_value=1e-3, gamma=0.99,
                     epsilon=0.2, use_lambda=True):
            """
            Initialize Lambda-PPO agent.

            Args:
                state_dim: Dimension of state space
                action_dim: Number of discrete actions
                hidden_dim: Hidden layer size
                lr_policy: Learning rate for policy
                lr_value: Learning rate for value function
                gamma: Discount factor
                epsilon: PPO clipping parameter
                use_lambda: Whether to use Lambda weighting (False = standard PPO)
            """
            self.gamma = gamma
            self.epsilon = epsilon
            self.use_lambda = use_lambda

            # Networks
            self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
            self.value = ValueNetwork(state_dim, hidden_dim)

            # Optimizers
            self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
            self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr_value)

        def select_action(self, state):
            """
            Select action from current policy.

            Args:
                state: Current state (numpy array)

            Returns:
                action: Selected action (int)
                log_prob: Log probability of action
            """
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                dist = self.policy.get_distribution(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            return action.item(), log_prob.item()

        def distribution_to_bivector(self, logits):
            """
            Convert action distribution to bivector representation.

            For discrete distribution, use:
            - Mean: μ = Σᵢ i * p(i)
            - Std: σ = sqrt(Σᵢ (i - μ)² * p(i))

            Then map to bivector: B[0] = μ, B[3] = σ

            Args:
                logits: Tensor of shape (batch, action_dim)

            Returns:
                bivectors: Tensor of shape (batch, 6)
            """
            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Action indices
            indices = torch.arange(probs.shape[-1], dtype=torch.float32)

            # Mean: μ = Σᵢ i * p(i)
            mu = torch.sum(probs * indices, dim=-1)

            # Variance: σ² = Σᵢ (i - μ)² * p(i)
            indices_expanded = indices.unsqueeze(0).expand_as(probs)
            mu_expanded = mu.unsqueeze(-1).expand_as(probs)
            variance = torch.sum(probs * (indices_expanded - mu_expanded)**2, dim=-1)
            sigma = torch.sqrt(variance + 1e-8)

            # Create bivector: [μ, 0, 0, σ, 0, 0]
            batch_size = logits.shape[0]
            bivectors = torch.zeros(batch_size, 6)
            bivectors[:, 0] = mu
            bivectors[:, 3] = sigma

            return bivectors

        def compute_policy_lambda(self, old_logits, new_logits):
            """
            Compute Λ between old and new policy distributions.

            Args:
                old_logits: Logits from old policy (batch, action_dim)
                new_logits: Logits from new policy (batch, action_dim)

            Returns:
                Lambda: Tensor of shape (batch,)
            """
            # Convert to bivectors
            B_old = self.distribution_to_bivector(old_logits)
            B_new = self.distribution_to_bivector(new_logits)

            # Compute commutator for each pair
            batch_size = old_logits.shape[0]
            Lambda = torch.zeros(batch_size)

            for i in range(batch_size):
                # Extract bivector components
                b1 = BivectorCl31(B_old[i].detach().numpy())
                b2 = BivectorCl31(B_new[i].detach().numpy())

                # Commutator
                comm = b1.commutator(b2)

                # Norm
                Lambda[i] = comm.norm()

            return Lambda

        def compute_advantages(self, states, actions, rewards, dones, old_logits):
            """
            Compute Lambda-weighted advantages.

            Args:
                states: Tensor (batch, state_dim)
                actions: Tensor (batch,)
                rewards: Tensor (batch,)
                dones: Tensor (batch,)
                old_logits: Tensor (batch, action_dim) - logits from old policy

            Returns:
                advantages: Tensor (batch,)
                returns: Tensor (batch,)
                Lambda: Tensor (batch,)
            """
            # Get value predictions
            values = self.value(states).detach()

            # Calculate returns and advantages using GAE
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)

            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]

                # TD error
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

                # GAE
                gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]

            # Calculate Lambda if enabled
            if self.use_lambda:
                # Get new logits
                new_logits = self.policy(states)

                # Compute Lambda
                Lambda = self.compute_policy_lambda(old_logits, new_logits)

                # Weight advantages by confidence
                confidence = torch.exp(-Lambda**2)
                weighted_advantages = advantages * confidence

                return weighted_advantages, returns, Lambda
            else:
                # Standard PPO (no Lambda weighting)
                Lambda = torch.zeros_like(advantages)
                return advantages, returns, Lambda

        def update(self, states, actions, rewards, dones, old_log_probs, old_logits, epochs=10):
            """
            PPO update with Lambda weighting.

            Args:
                states: Tensor (batch, state_dim)
                actions: Tensor (batch,)
                rewards: Tensor (batch,)
                dones: Tensor (batch,)
                old_log_probs: Tensor (batch,)
                old_logits: Tensor (batch, action_dim)
                epochs: Number of optimization epochs

            Returns:
                Dictionary of training metrics
            """
            # Compute advantages
            advantages, returns, Lambda = self.compute_advantages(
                states, actions, rewards, dones, old_logits
            )

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Multiple epochs of optimization
            metrics = {
                'policy_loss': [],
                'value_loss': [],
                'mean_lambda': []
            }

            for epoch in range(epochs):
                # Policy loss (PPO clipped objective)
                new_log_probs = self.policy.log_prob(states, actions)
                ratio = torch.exp(new_log_probs - old_log_probs)

                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

                policy_loss = -torch.min(
                    ratio * advantages,
                    clipped_ratio * advantages
                ).mean()

                # Value loss
                predicted_values = self.value(states)
                value_loss = F.mse_loss(predicted_values, returns)

                # Update policy
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer_policy.step()

                # Update value
                self.optimizer_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.optimizer_value.step()

                # Track metrics
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['mean_lambda'].append(Lambda.mean().item())

            # Return average metrics
            return {
                'policy_loss': np.mean(metrics['policy_loss']),
                'value_loss': np.mean(metrics['value_loss']),
                'mean_lambda': np.mean(metrics['mean_lambda'])
            }


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

if TORCH_AVAILABLE:
    def collect_trajectories(env, agent, n_steps=2048):
        """
        Collect trajectories from environment.

        Args:
            env: Gymnasium environment
            agent: LambdaPPO agent
            n_steps: Number of steps to collect

        Returns:
            Dictionary of trajectories
        """
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        logits = []

        state, _ = env.reset()
        episode_rewards = []
        current_episode_reward = 0

        for step in range(n_steps):
            # Select action
            action, log_prob = agent.select_action(state)

            # Get logits for Lambda calculation
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logit = agent.policy(state_tensor)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            logits.append(logit.squeeze(0))

            current_episode_reward += reward

            if done:
                state, _ = env.reset()
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
            else:
                state = next_state

        return {
            'states': torch.FloatTensor(np.array(states)),
            'actions': torch.LongTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'dones': torch.FloatTensor(dones),
            'log_probs': torch.FloatTensor(log_probs),
            'logits': torch.stack(logits),
            'episode_rewards': episode_rewards
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_cartpole():
    """
    Example: Lambda-PPO on CartPole-v1

    Requires: pip install gymnasium
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        return

    try:
        import gymnasium as gym
    except ImportError:
        print("Gymnasium not available. Install with: pip install gymnasium")
        return

    print("="*80)
    print("LAMBDA-PPO EXAMPLE: CartPole-v1")
    print("="*80)
    print()

    env = gym.make('CartPole-v1')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print()

    # Create agent
    agent = LambdaPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        use_lambda=True
    )

    print("Training Lambda-PPO for 100 iterations...")
    print()

    for iteration in range(100):
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

        # Report
        if len(trajectories['episode_rewards']) > 0:
            avg_reward = np.mean(trajectories['episode_rewards'])
            print(f"Iteration {iteration+1:3d}: "
                  f"Avg Reward = {avg_reward:6.1f}, "
                  f"Policy Loss = {metrics['policy_loss']:.4f}, "
                  f"Mean Lambda = {metrics['mean_lambda']:.4f}")

            # Solved if average reward > 195
            if avg_reward > 195:
                print()
                print(f"[SOLVED] CartPole solved in {iteration+1} iterations!")
                break

    env.close()


def main():
    """Main function"""
    print("\n")
    print("="*80)
    print("LAMBDA-PPO: STARTER IMPLEMENTATION")
    print("="*80)
    print()

    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch not installed!")
        print()
        print("Lambda-PPO requires PyTorch.")
        print("Install with: pip install torch")
        print()
        return

    print("Lambda-PPO is ready to use!")
    print()
    print("Key classes:")
    print("  - PolicyNetwork: state -> action logits")
    print("  - ValueNetwork: state -> value estimate")
    print("  - LambdaPPO: Main agent with Lambda weighting")
    print()
    print("Run example_cartpole() to test on CartPole-v1")
    print()

    # Run example if gymnasium available
    try:
        import gymnasium as gym
        print("Running CartPole example...")
        print()
        example_cartpole()
    except ImportError:
        print("Gymnasium not installed. Install with: pip install gymnasium")
        print("Then run: python lambda_ppo_starter.py")


if __name__ == "__main__":
    main()
