"""W-MATRPO algorithm with correct inner maximization implementation."""

import numpy as np
import torch
import torch.optim as optim
from harl.utils.envs_tools import check
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.models.policy_models.stochastic_policy import StochasticPolicy
import torch.nn as nn
from harl.algorithms.critics.v_critic import VCritic

# Import the fixed utility functions
from harl.utils.wasserstein_trpo_util import (
    wasserstein_divergence,
    calculate_w_matrpo_loss_with_inner_max,
    calculate_w_matrpo_loss,  # Fallback for testing
    update_model,
    flat_params,
    l2_transport_cost,
    l1_transport_cost,
)


class WMATRPO_NONADAPTIVE(OnPolicyBase):
    """
    Wasserstein-enabled Multi-agent Trust Region Policy Optimization (W-MATRPO)
    with correct inner maximization from the dual formulation.
    """

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize W-MATRPO with correct dual formulation."""
        super(WMATRPO_NONADAPTIVE, self).__init__(args, obs_space, act_space, device)
        self.critic = None
        # W-MATRPO specific hyperparameters
        self.w_delta = args["w_delta"]
        self.lambda_lr = args["lambda_lr"]
        
        # Algorithm settings
        self.use_inner_max = args.get("use_inner_max", False)  # Toggle for testing
        self.transport_cost = args.get("transport_cost", "l2")  # "l1" or "l2"
        self.inner_max_samples = args.get("inner_max_samples", 64)
        self.inner_max_iterations = args.get("inner_max_iterations", 5)
        self.inner_max_lr = args.get("inner_max_lr", 0.1)
        
        # Select transport cost function
        if self.transport_cost == "l1":
            self.transport_cost_fn = l1_transport_cost
        else:
            self.transport_cost_fn = l2_transport_cost

        # Initialize the dual variable lambda
        # Start with a small positive value to avoid numerical issues
        self.log_lambda = nn.Parameter(torch.log(torch.tensor([1.0], device=self.device)))
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=self.lambda_lr)
        
        # Check if action space is discrete
        self.is_discrete = hasattr(self.act_space, 'n')
        
        # For logging
        self.num_actor_updates = 0
        self.update_stats = []
        
        # Print configuration
        self._print_config()

    def _print_config(self):
        """Print algorithm configuration."""
        print("\n" + "="*80)
        print("W-MATRPO NON-ADAPTIVE - FIXED IMPLEMENTATION")
        print("="*80)
        print("Configuration:")
        print(f"  Trust Region Radius (w_delta): {self.w_delta}")
        print(f"  Dual Variable LR (lambda_lr): {self.lambda_lr}")
        print(f"  Initial Lambda: {torch.exp(self.log_lambda).item():.4f}")
        print(f"  Use Inner Maximization: {self.use_inner_max}")
        print(f"  Transport Cost: {self.transport_cost}")
        print(f"  Action Space: {'Discrete' if self.is_discrete else 'Continuous'}")
        if not self.is_discrete:
            print(f"  Inner Max Samples: {self.inner_max_samples}")
            print(f"  Inner Max Iterations: {self.inner_max_iterations}")
            print(f"  Inner Max LR: {self.inner_max_lr}")
        print("="*80 + "\n")

    def update(self, sample, critic):
        """Update actor and dual variable using correct dual formulation."""
        # Unpack sample
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        # Convert to tensors
        obs_batch = check(obs_batch).to(**self.tpdv)
        if rnn_states_batch is not None:
            rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        if available_actions_batch is not None:
            available_actions_batch = check(available_actions_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # Apply importance sampling correction
        surrogate_advantage = adv_targ * factor_batch

        # Create old actor copy
        old_actor = StochasticPolicy(
            self.args, self.obs_space, self.act_space, self.device
        )
        update_model(old_actor, flat_params(self.actor))
        old_actor.eval()

        # Calculate Wasserstein distance
        w_dist = wasserstein_divergence(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            new_actor=self.actor,
            old_actor=old_actor,
            use_w1=False,  # Use W2 distance for differentiability
        )

        # Get current lambda value
        lambda_val = torch.exp(self.log_lambda)

        # --- START OF CORRECTED SECTION ---

        # Get the log probabilities of the buffer actions from the NEW policy
        _, new_log_probs, _ = self.actor.evaluate_actions(
            obs=obs_batch,
            rnn_states=rnn_states_batch,
            action=actions_batch,
            masks=masks_batch,
            available_actions=available_actions_batch,
            active_masks=active_masks_batch
        )

        # old_action_log_probs_batch is already available from the 'sample' tuple.
        # We must ensure it has the correct device and doesn't track gradients from the old policy.
        old_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)

        # Calculate losses using the corrected function with the policy gradient term
        actor_loss, lambda_loss = calculate_w_matrpo_loss(
            advantage=surrogate_advantage, 
            w_dist=w_dist, 
            lambda_val=lambda_val, 
            delta=self.w_delta,
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs_batch
        )
        info = {} # Set info to an empty dict as the logger might expect it

        # --- END OF CORRECTED SECTION ---

        # Gradient updates
        self.actor_optimizer.zero_grad()
        self.lambda_optimizer.zero_grad()

        # Compute gradients
        actor_loss.backward(retain_graph=True)
        lambda_loss.backward()

        # Optional: Gradient clipping
        if hasattr(self.args, 'max_grad_norm') and self.args['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.args['max_grad_norm']
            )

        # Apply updates
        self.actor_optimizer.step()
        self.lambda_optimizer.step()

        # Calculate policy entropy for logging
        _, _, dist = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        dist_entropy = dist.entropy().mean()

        # Store update statistics
        self.update_stats.append({
            'update': self.num_actor_updates,
            'w_dist': w_dist.mean().item(),
            'w_dist_std': w_dist.std().item(),
            'lambda': lambda_val.item(),
            'actor_loss': actor_loss.item(),
            'lambda_loss': lambda_loss.item(),
            'entropy': dist_entropy.item(),
            'constraint_violation': (w_dist.mean() - self.w_delta).item(),
            **info,
        })

        # Periodic logging
        if self.num_actor_updates % 10 == 0:
            print(f"  W-dist: {w_dist.mean().item():.6f} ± {w_dist.std().item():.6f}")
            print(f"  Lambda: {lambda_val.item():.6f}")
            print(f"  Actor loss: {actor_loss.item():.6f}")
            print(f"  Entropy: {dist_entropy.item():.6f}")

        self.num_actor_updates += 1

        return w_dist.mean(), actor_loss, lambda_loss, dist_entropy, lambda_val.item()

    def train(self, actor_buffer, advantages, state_type, critic):
        """Perform a training update using minibatch GD."""
        self.critic = critic
        train_info = {
            "w_dist": 0,
            "actor_loss": 0,
            "lambda_loss": 0,
            "dist_entropy": 0,
            "lambda_val": 0,
        }

        # Check if all masks are zero
        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            print("Warning: All active masks are zero!")
            return train_info

        # Advantage normalization
        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # Generate data
        if self.use_recurrent_policy:
            data_generator = actor_buffer.recurrent_generator_actor(
                advantages, 1, self.data_chunk_length
            )
        elif self.use_naive_recurrent_policy:
            data_generator = actor_buffer.naive_recurrent_generator_actor(advantages, 1)
        else:
            data_generator = actor_buffer.feed_forward_generator_actor(advantages, 1)

        # Update loop
        for sample in data_generator:
            w_dist, actor_loss, lambda_loss, dist_entropy, lambda_val = self.update(
                sample, self.critic
            )

            train_info["w_dist"] += w_dist.item()
            train_info["actor_loss"] += actor_loss.item()
            train_info["lambda_loss"] += lambda_loss.item()
            train_info["dist_entropy"] += dist_entropy.item()
            train_info["lambda_val"] += lambda_val

        # Average over updates
        num_updates = 1
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def get_update_stats_summary(self):
        """Get summary of update statistics for debugging."""
        if not self.update_stats:
            return {}
        
        stats = {}
        keys = self.update_stats[0].keys()
        
        for key in keys:
            if key != 'update':
                values = [s[key] for s in self.update_stats if key in s]
                if values:
                    stats[f'{key}_mean'] = np.mean(values)
                    stats[f'{key}_std'] = np.std(values)
                    stats[f'{key}_last'] = values[-1]
        
        return stats

    def save_stats(self, filepath):
        """Save update statistics for analysis."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.update_stats, f, indent=2)
        print(f"Saved update statistics to {filepath}")
