"""
W-MATRPO Utility Functions

This file provides the core components for implementing the Wasserstein-enabled
Multi-agent Trust Region Policy Optimization (W-MATRPO) algorithm.
"""

# ==============================================================================
# How to Integrate This File into Your Training Loop
# ==============================================================================
#
# This file contains UTILITY functions. Your main training script will import
# and call these functions. The general workflow is as follows:
#
# 1. SETUP:
#    - Initialize N actor networks, 1 centralized critic, and their optimizers.
#    - For each agent `i`, initialize a dual variable `lambda_i = torch.tensor([1.0], requires_grad=True)`.
#    - For each `lambda_i`, create a separate optimizer (e.g., Adam or SGD).
#
# 2. DATA COLLECTION:
#    - In a loop, have agents interact with the environment to collect
#      trajectories (obs, actions, rewards, etc.) into a buffer.
#
# 3. UPDATE PHASE (LOOPING SEQUENTIALLY THROUGH AGENTS):
#    For each agent `i`:
#      a. From the buffer, get the collected data.
#      b. Calculate advantage estimates A(s, a) using the critic.
#
#      c. Compute the Wasserstein distance trust region constraint:
#         `w_dist = wasserstein_divergence(...)`
#
#      d. Compute the actor and lambda losses using the dual formulation:
#         `actor_loss, lambda_loss = calculate_w_matrpo_loss(advantage, w_dist, lambda_i, delta_i)`
#
#      e. Perform backpropagation and update the networks:
#         - Update actor `i` using `actor_loss`.
#         - Update dual variable `lambda_i` using `lambda_loss`.
#         - Clamp `lambda_i` to be non-negative after the update.
#
#      f. Update the centralized critic by minimizing the TD-error.
#
# ==============================================================================

import torch
import ot

# ==============================================================================
# General PyTorch Model Utilities
# ==============================================================================

def flat_params(model):
    """
    Flattens the parameters of a PyTorch model into a single 1D tensor.
    Role: General utility.
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    """
    Updates a model's parameters from a flattened 1D tensor.
    Role: General utility.
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


def wasserstein_divergence(
    obs,
    rnn_states,
    action,
    masks,
    available_actions,
    active_masks,
    new_actor,
    old_actor,
):
    """
    Calculates the Wasserstein distance between the old and new actor policies.
    Role: This is called once per agent update (step 3c in the guide above)
          to compute the value of the trust region constraint. The result is
          fed into `calculate_w_matrpo_loss`.
    """
    # Get the new and old policy distributions from the actor networks
    _, _, new_dist = new_actor.evaluate_actions(
        obs, rnn_states, action, masks, available_actions, active_masks
    )
    with torch.no_grad():
        _, _, old_dist = old_actor.evaluate_actions(
            obs, rnn_states, action, masks, available_actions, active_masks
        )

    # Dispatch to the correct helper based on the action space type
    if new_dist.__class__.__name__ == "FixedCategorical": # Discrete actions
        w_dist = wasserstein_pot(old_dist.logits, new_dist.logits)
    else: # Continuous actions
        w_dist = _wasserstein_normal_normal(old_dist, new_dist)

    # Reshape for consistency
    if len(w_dist.shape) == 1:
        w_dist = w_dist.unsqueeze(1)
    return w_dist

# ==============================================================================
# W-MATRPO Dual Formulation Loss Calculation
# This is the core of the W-MATRPO algorithm implementation.
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
