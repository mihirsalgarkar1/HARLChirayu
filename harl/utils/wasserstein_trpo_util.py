"""
<<<<<<< HEAD
W-MATRPO Utility Functions

This file provides the core components for implementing the Wasserstein-enabled
Multi-agent Trust Region Policy Optimization (W-MATRPO) algorithm.
"""
import torch
import ot

=======
W-MATRPO Utility Functions - Fixed Implementation

This file provides the corrected core components for implementing the Wasserstein-enabled
Multi-agent Trust Region Policy Optimization (W-MATRPO) algorithm, including the
crucial inner maximization step from the dual formulation.
"""

import torch
import torch.nn.functional as F
import ot
import numpy as np
from typing import Tuple, Optional, Callable

# ==============================================================================
# General PyTorch Model Utilities
# ==============================================================================
>>>>>>> b1e3fc3 (potential updates to trial setup)

def flat_params(model):
    """
    Flattens the parameters of a PyTorch model into a single 1D tensor.
<<<<<<< HEAD
    Role: General utility.
=======
>>>>>>> b1e3fc3 (potential updates to trial setup)
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    """
    Updates a model's parameters from a flattened 1D tensor.
<<<<<<< HEAD
    Role: General utility.
=======
>>>>>>> b1e3fc3 (potential updates to trial setup)
    """
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index : index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length

# ==============================================================================
# Wasserstein Distance Calculation
# These functions compute the W-distance, which serves as the trust region.
# ==============================================================================

<<<<<<< HEAD
def _wasserstein_normal_normal(p, q):
    """
    Computes the squared Wasserstein-2 distance between two Normal distributions
    with diagonal covariance matrices (for continuous action spaces).
    Role: A helper function called by `wasserstein_divergence`.
    """
    p_loc = p.loc.to(torch.float64)
    q_loc = q.loc.to(torch.float64)
    p_scale = p.scale.to(torch.float64)
    q_scale = q.scale.to(torch.float64)
    mean_diff_sq = (p_loc - q_loc).pow(2).sum(-1)
    scale_diff_sq = (p_scale - q_scale).pow(2).sum(-1)
    return mean_diff_sq + scale_diff_sq


def wasserstein_pot(p_logits, q_logits):
    """
    Computes the Wasserstein-1 distance for a batch of discrete distributions
    using the POT library (for discrete action spaces).
    Role: A helper function called by `wasserstein_divergence`.
    """
    p_probs = torch.softmax(p_logits, dim=-1)
    q_probs = torch.softmax(q_logits, dim=-1)
    n_categories = p_logits.shape[-1]
    cost_matrix = torch.abs(
        torch.arange(n_categories, device=p_logits.device, dtype=torch.float32).unsqueeze(1) -
        torch.arange(n_categories, device=p_logits.device, dtype=torch.float32).unsqueeze(0)
    )
    p_probs_np = p_probs.detach().cpu().numpy()
    q_probs_np = q_probs.detach().cpu().numpy()
    cost_matrix_np = cost_matrix.cpu().numpy()
    w_distances = [ot.emd2(p_probs_np[i], q_probs_np[i], cost_matrix_np) for i in range(p_probs.shape[0])]
    return torch.tensor(w_distances, device=p_logits.device, dtype=torch.float32)
=======
# ==============================================================================
# Transport Cost Functions
# ==============================================================================

def l1_transport_cost(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """
    L1 (Manhattan) distance transport cost.
    
    Args:
        a1: Actions of shape (..., action_dim)
        a2: Actions of shape (..., action_dim)
    
    Returns:
        Cost of shape (...,)
    """
    return torch.abs(a1 - a2).sum(dim=-1)


def l2_transport_cost(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """
    L2 (Euclidean) distance transport cost.
    
    Args:
        a1: Actions of shape (..., action_dim)
        a2: Actions of shape (..., action_dim)
    
    Returns:
        Cost of shape (...,)
    """
    return torch.norm(a1 - a2, p=2, dim=-1)


def discrete_transport_cost(a1: torch.Tensor, a2: torch.Tensor, 
                           n_actions: int) -> torch.Tensor:
    """
    Transport cost for discrete actions (assumes ordered actions).
    
    Args:
        a1: Action indices of shape (...,)
        a2: Action indices of shape (...,)
        n_actions: Total number of actions
    
    Returns:
        Cost of shape (...,)
    """
    return torch.abs(a1.float() - a2.float())


# ==============================================================================
# Wasserstein Distance Calculation (Corrected)
# ==============================================================================

def wasserstein_1_continuous(old_dist, new_dist, n_samples: int = 100):
    """
    Approximates W1 distance between continuous distributions using sampling.
    
    For diagonal Gaussians, we can use the fact that W1 <= W2, and for 
    numerical stability, we approximate using samples.
    """
    # Sample from both distributions
    old_samples = old_dist.sample((n_samples,))  # (n_samples, batch_size, action_dim)
    new_samples = new_dist.sample((n_samples,))
    
    # Compute empirical W1 using POT
    batch_size = old_samples.shape[1]
    w_dists = []
    
    for b in range(batch_size):
        old_batch = old_samples[:, b, :].detach()
        new_batch = new_samples[:, b, :].detach()
        
        # Cost matrix
        cost_matrix = torch.cdist(old_batch, new_batch, p=1)  # L1 distance
        
        # Uniform weights
        a = torch.ones(n_samples, device=old_batch.device) / n_samples
        b = torch.ones(n_samples, device=new_batch.device) / n_samples
        
        # Solve optimal transport
        cost_matrix_np = cost_matrix.cpu().numpy()
        a_np = a.cpu().numpy()
        b_np = b.cpu().numpy()
        
        w_dist = ot.emd2(a_np, b_np, cost_matrix_np)
        w_dists.append(w_dist)
    
    return torch.tensor(w_dists, device=old_dist.loc.device, dtype=torch.float32)


def wasserstein_1_discrete(old_logits: torch.Tensor, new_logits: torch.Tensor):
    """
    Computes the Wasserstein-1 distance for discrete distributions.
    """
    old_probs = torch.softmax(old_logits, dim=-1)
    new_probs = torch.softmax(new_logits, dim=-1)
    
    batch_size, n_actions = old_probs.shape
    
    # Cost matrix (assumes ordered actions)
    cost_matrix = torch.abs(
        torch.arange(n_actions, device=old_probs.device, dtype=torch.float32).unsqueeze(1) -
        torch.arange(n_actions, device=old_probs.device, dtype=torch.float32).unsqueeze(0)
    )
    
    # Solve optimal transport for each batch element
    old_probs_np = old_probs.detach().cpu().numpy()
    new_probs_np = new_probs.detach().cpu().numpy()
    cost_matrix_np = cost_matrix.cpu().numpy()
    
    w_dists = []
    for b in range(batch_size):
        w_dist = ot.emd2(old_probs_np[b], new_probs_np[b], cost_matrix_np)
        w_dists.append(w_dist)
    
    return torch.tensor(w_dists, device=old_probs.device, dtype=torch.float32)
>>>>>>> b1e3fc3 (potential updates to trial setup)


def wasserstein_divergence(
    obs,
    rnn_states,
    action,
    masks,
    available_actions,
    active_masks,
    new_actor,
    old_actor,
    use_w1: bool = True,
):
    """
<<<<<<< HEAD
    Calculates the Wasserstein distance between the old and new actor policies.
    Role: This is called once per agent update (step 3c in the guide above)
          to compute the value of the trust region constraint. The result is
          fed into `calculate_w_matrpo_loss`.
    """
    # Get the new and old policy distributions from the actor networks
=======
    Calculates the Wasserstein distance between old and new policies.
    
    Args:
        use_w1: If True, use W1 distance. If False, fall back to W2 for continuous.
    """
    # Get policy distributions
>>>>>>> b1e3fc3 (potential updates to trial setup)
    _, _, new_dist = new_actor.evaluate_actions(
        obs, rnn_states, action, masks, available_actions, active_masks
    )
    
    with torch.no_grad():
        _, _, old_dist = old_actor.evaluate_actions(
            obs, rnn_states, action, masks, available_actions, active_masks
        )
<<<<<<< HEAD

    # Dispatch to the correct helper based on the action space type
    if new_dist.__class__.__name__ == "FixedCategorical": # Discrete actions
        w_dist = wasserstein_pot(old_dist.logits, new_dist.logits)
    else: # Continuous actions
        w_dist = _wasserstein_normal_normal(old_dist, new_dist)
=======
    
    # Compute Wasserstein distance based on action space
    if hasattr(new_dist, 'logits'):  # Discrete actions
        w_dist = wasserstein_1_discrete(old_dist.logits, new_dist.logits)
    else:  # Continuous actions
        if use_w1:
            w_dist = wasserstein_1_continuous(old_dist, new_dist)
        else:
            # Fall back to W2 for numerical stability if needed
            w_dist = _wasserstein_2_diagonal_gaussians(old_dist, new_dist)
    
    # Ensure consistent shape
    if len(w_dist.shape) == 1:
        w_dist = w_dist.unsqueeze(1)
    
    # Apply active masks
    w_dist = w_dist * active_masks
    
    return w_dist
>>>>>>> b1e3fc3 (potential updates to trial setup)

    # Reshape for consistency
    if len(w_dist.shape) == 1:
        w_dist = w_dist.unsqueeze(1)
    return w_dist

<<<<<<< HEAD
# ==============================================================================
# W-MATRPO Dual Formulation Loss Calculation
# ==============================================================================

def calculate_w_matrpo_loss(advantage, w_dist, lambda_val, delta):
    """
    Calculates the actor and lambda losses based on the dual formulation.
    Role: This function implements the core logic of your paper's Equation (6).
          It's called once per agent update (step 3d in the guide above). It takes
          the advantage, the computed Wasserstein distance, the dual variable (lambda),
          and the trust region size (delta) as input.

    Args:
        advantage (torch.Tensor): The advantage estimate M_i(s, a).
        w_dist (torch.Tensor): The computed Wasserstein distance from `wasserstein_divergence`.
        lambda_val (torch.Tensor): The agent's current dual variable lambda.
        delta (float): The agent's trust region radius delta.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - actor_loss: The loss for updating the policy network.
            - lambda_loss: The loss for updating the dual variable.
    """
    # This is the loss for the policy (actor) network.
    # It tries to maximize the advantage, while the second term acts as a
    # penalty/incentive based on the constraint satisfaction.
    # We detach `lambda_val` because the actor's gradient shouldn't flow into lambda.
    
    actor_loss = -advantage.mean() + lambda_val.detach() * (w_dist.mean() - delta)

    # This is the loss for the dual variable `lambda`.
    # Minimizing this loss via gradient descent performs gradient ascent on lambda.
    # This update rule pushes lambda higher if the constraint is violated (w_dist > delta)
    # and lower if it's satisfied, effectively enforcing the trust region.
    
    lambda_loss = -(lambda_val * (w_dist.mean() - delta))

    return actor_loss, lambda_loss
=======
def _wasserstein_2_diagonal_gaussians(p, q):
    """
    Closed-form W2 distance for diagonal Gaussians (as fallback).
    """
    mean_diff_sq = (p.loc - q.loc).pow(2).sum(-1)
    scale_diff_sq = (p.scale - q.scale).pow(2).sum(-1)
    return torch.sqrt(mean_diff_sq + scale_diff_sq + 1e-8)


# ==============================================================================
# Inner Maximization for Dual Formulation
# ==============================================================================

def solve_inner_maximization_continuous(
    obs: torch.Tensor,
    rnn_states: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
    available_actions: Optional[torch.Tensor],
    active_masks: torch.Tensor,
    old_actor,
    critic,
    lambda_val: torch.Tensor,
    transport_cost_fn: Callable = l2_transport_cost,
    n_samples: int = 64,
    n_iterations: int = 5,
    lr: float = 0.1,
):
    """
    Solves the inner maximization problem for continuous actions:
    max_{a'} [A^{π_old}(s, a') - λ * c(a, a')]
    
    Uses gradient ascent on sampled actions.
    """
    batch_size = obs.shape[0]
    
    with torch.no_grad():
        # Get old policy distribution
        # Get old policy distribution
        _, _, old_dist = old_actor.evaluate_actions(
        obs=obs,
        rnn_states=rnn_states,
        action=actions,
        masks=masks,
        available_actions=available_actions,
        active_masks=active_masks
    )
        # Sample initial actions from old policy
        sampled_actions = old_dist.sample((n_samples,))  # (n_samples, batch_size, action_dim)
        sampled_actions = sampled_actions.transpose(0, 1)  # (batch_size, n_samples, action_dim)
        
        # Also include the mean action
        mean_actions = old_dist.loc.unsqueeze(1)  # (batch_size, 1, action_dim)
        sampled_actions = torch.cat([mean_actions, sampled_actions], dim=1)
        n_samples += 1
    
    # Make actions require grad for optimization
    opt_actions = sampled_actions.clone().detach().requires_grad_(True)
    
    # Gradient ascent to refine actions
    for _ in range(n_iterations):
        # Reshape for critic evaluation
        opt_actions_flat = opt_actions.reshape(-1, opt_actions.shape[-1])
        obs_expanded = obs.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, obs.shape[-1])
        
        if rnn_states is not None:
            rnn_states_expanded = rnn_states.unsqueeze(1).expand(-1, n_samples, -1, -1)
            rnn_states_expanded = rnn_states_expanded.reshape(-1, *rnn_states.shape[1:])
        else:
            rnn_states_expanded = None
        
        masks_expanded = masks.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, masks.shape[-1])
        
        # Get Q-values for all sampled actions
        with torch.no_grad():
            q_values = critic.get_values(obs_expanded, rnn_states_expanded, masks_expanded, opt_actions_flat)
            q_values = q_values.reshape(batch_size, n_samples)
            
            # Get V-values for advantage calculation
            v_values = critic.get_values(obs, rnn_states, masks)
        
        # Calculate advantages
        advantages = q_values - v_values
        
        # Calculate transport costs from old policy mean
        old_actions = old_dist.loc.unsqueeze(1).expand(-1, n_samples, -1)
        transport_costs = transport_cost_fn(old_actions, opt_actions)
        
        # Objective: A(s, a') - λ * c(a, a')
        objective = advantages - lambda_val * transport_costs
        
        # Gradient ascent step
        if opt_actions.grad is not None:
            opt_actions.grad.zero_()
        objective.sum().backward()
        
        with torch.no_grad():
            opt_actions += lr * opt_actions.grad
            
            # Optional: Clip actions to valid range
            if hasattr(old_actor, 'action_range'):
                opt_actions.clamp_(old_actor.action_range[0], old_actor.action_range[1])
    
    # Select best action for each batch element
    with torch.no_grad():
        # Final evaluation
        opt_actions_flat = opt_actions.reshape(-1, opt_actions.shape[-1])
        q_values = critic.get_values(obs_expanded, rnn_states_expanded, masks_expanded, opt_actions_flat)
        q_values = q_values.reshape(batch_size, n_samples)
        advantages = q_values - v_values
        
        transport_costs = transport_cost_fn(old_dist.loc.unsqueeze(1).expand(-1, n_samples, -1), opt_actions)
        objective = advantages - lambda_val * transport_costs
        
        # Get best action index for each batch element
        best_indices = objective.argmax(dim=1)
        best_actions = opt_actions[torch.arange(batch_size), best_indices]
        best_advantages = advantages[torch.arange(batch_size), best_indices]
        best_objectives = objective[torch.arange(batch_size), best_indices]
    
    return best_actions, best_advantages, best_objectives


def solve_inner_maximization_discrete(
    obs: torch.Tensor,
    rnn_states: torch.Tensor,
    masks: torch.Tensor,
    available_actions: Optional[torch.Tensor],
    active_masks: torch.Tensor,
    old_actor,
    critic,
    lambda_val: torch.Tensor,
    n_actions: int,
):
    """
    Solves the inner maximization problem for discrete actions:
    max_{a'} [A^{π_old}(s, a') - λ * c(a, a')]
    
    For discrete actions, we can evaluate all possible actions.
    """
    batch_size = obs.shape[0]
    
    with torch.no_grad():
        # Get old policy distribution
        old_actor_features = old_actor.base(obs, rnn_states, masks)
        old_logits = old_actor.dist.logits
        old_probs = torch.softmax(old_logits, dim=-1)
        
        # Most likely old action
        old_actions = old_probs.argmax(dim=-1)
        
        # Evaluate Q-values for all possible actions
        all_q_values = []
        for a in range(n_actions):
            action_batch = torch.full((batch_size,), a, device=obs.device, dtype=torch.long)
            q_val = critic.get_values(obs, rnn_states, masks, action_batch)
            all_q_values.append(q_val)
        
        all_q_values = torch.stack(all_q_values, dim=1)  # (batch_size, n_actions)
        
        # Get V-values
        v_values = critic.get_values(obs, rnn_states, masks)
        
        # Calculate advantages for all actions
        all_advantages = all_q_values - v_values.unsqueeze(1)
        
        # Calculate transport costs for all actions
        old_actions_expanded = old_actions.unsqueeze(1).expand(-1, n_actions)
        all_actions = torch.arange(n_actions, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        transport_costs = discrete_transport_cost(old_actions_expanded, all_actions, n_actions)
        
        # Objective for all actions
        objectives = all_advantages - lambda_val * transport_costs
        
        # Apply available actions mask if provided
        if available_actions is not None:
            objectives = objectives.masked_fill(~available_actions.bool(), -float('inf'))
        
        # Get best action
        best_indices = objectives.argmax(dim=1)
        best_advantages = all_advantages[torch.arange(batch_size), best_indices]
        best_objectives = objectives[torch.arange(batch_size), best_indices]
    
    return best_indices, best_advantages, best_objectives


# ==============================================================================
# W-MATRPO Loss Calculation with Inner Maximization
# ==============================================================================

def calculate_w_matrpo_loss_with_inner_max(
    obs: torch.Tensor,
    rnn_states: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
    available_actions: Optional[torch.Tensor],
    active_masks: torch.Tensor,
    advantages: torch.Tensor,
    w_dist: torch.Tensor,
    lambda_val: torch.Tensor,
    delta: float,
    new_actor,
    old_actor,
    critic,
    is_discrete: bool,
    transport_cost_fn: Callable = l2_transport_cost,
):
    """
    Calculates actor and lambda losses using the dual formulation with inner maximization.
    
    This implements the full dual formulation from equation (17) in the paper.
    """
    # Solve inner maximization problem
    if is_discrete:
        n_actions = new_actor.act.action_out.out_features
        best_actions, best_advantages, best_objectives = solve_inner_maximization_discrete(
            obs, rnn_states, masks, available_actions, active_masks,
            old_actor, critic, lambda_val, n_actions
        )
    else:
        best_actions, best_advantages, best_objectives = solve_inner_maximization_continuous(
            obs, rnn_states, actions, masks, available_actions, active_masks,
            old_actor, critic, lambda_val, transport_cost_fn
        )
    
    # The actor loss encourages the new policy to take actions similar to the
    # optimal actions found in the inner maximization
    if is_discrete:
        # For discrete actions, use cross-entropy to encourage selecting best_actions
        _, new_dist = new_actor(obs, rnn_states, masks, available_actions)
        actor_loss = F.cross_entropy(new_dist.logits, best_actions, reduction='none')
        actor_loss = (actor_loss * active_masks.squeeze(-1)).sum() / active_masks.sum()
    else:
        # For continuous actions, use MSE or negative log-likelihood
        _, new_dist = new_actor(obs, rnn_states, masks, available_actions)
        # Negative log-likelihood of best actions under new policy
        actor_loss = -new_dist.log_prob(best_actions).sum(dim=-1, keepdim=True)
        actor_loss = (actor_loss * active_masks).sum() / active_masks.sum()
    
    # Alternative formulation: Use the objective value directly
    # actor_loss = -best_objectives.mean()
    
    # Lambda loss remains the same - enforce trust region constraint
    constraint_violation = w_dist.mean() - delta
    lambda_loss = -lambda_val * constraint_violation
    
    # Return additional info for logging
    info = {
        'best_advantages': best_advantages.mean().item(),
        'best_objectives': best_objectives.mean().item(),
        'original_advantages': advantages.mean().item(),
        'constraint_violation': constraint_violation.item(),
    }
    
    return actor_loss, lambda_loss, info


# ==============================================================================
# Simplified Loss Function (Backward Compatible)
# ==============================================================================

def calculate_w_matrpo_loss(advantage, w_dist, lambda_val, delta, new_log_probs, old_log_probs):
    """
    Simplified loss calculation with the CORRECT policy gradient term.
    """
    # Calculate the importance sampling ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # The policy gradient loss
    pg_loss = -(ratio * advantage).mean()
    
    # The trust region constraint loss
    constraint_loss = lambda_val.detach() * (w_dist.mean() - delta)

    # The final actor loss
    actor_loss = pg_loss + constraint_loss
    
    # Lambda loss remains the same
    lambda_loss = -(lambda_val * (w_dist.mean() - delta))
    
    return actor_loss, lambda_loss

>>>>>>> b1e3fc3 (potential updates to trial setup)
