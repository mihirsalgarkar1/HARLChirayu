"""TRPO and W-MATRPO utility functions."""
import torch
import ot


def flat_grad(grads):
    """Flatten the gradients."""
    grad_flatten = []
    for grad in grads:
        if grad is None:
            continue
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    """Flatten the hessians."""
    hessians_flatten = []
    for hessian in hessians:
        if hessian is None:
            continue
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    """Flatten the parameters."""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    """Update the model parameters."""
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index : index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def kl_approx(p, q):
    """KL divergence between two distributions."""
    r = torch.exp(q - p)
    kl = r - 1 - q + p
    return kl


def _kl_normal_normal(p, q):
    """KL divergence between two normal distributions."""
    var_ratio = (p.scale.to(torch.float64) / q.scale.to(torch.float64)).pow(2)
    t1 = (
        (p.loc.to(torch.float64) - q.loc.to(torch.float64)) / q.scale.to(torch.float64)
    ).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def kl_divergence(
    obs,
    rnn_states,
    action,
    masks,
    available_actions,
    active_masks,
    new_actor,
    old_actor,
):
    """KL divergence between two distributions."""
    _, _, new_dist = new_actor.evaluate_actions(
        obs, rnn_states, action, masks, available_actions, active_masks
    )
    with torch.no_grad():
        _, _, old_dist = old_actor.evaluate_actions(
            obs, rnn_states, action, masks, available_actions, active_masks
        )
    if new_dist.__class__.__name__ == "FixedCategorical":  # discrete action
        new_logits = new_dist.logits
        old_logits = old_dist.logits
        kl = kl_approx(old_logits, new_logits)
    else:  # continuous action
        kl = _kl_normal_normal(old_dist, new_dist)

    if len(kl.shape) > 1:
        kl = kl.sum(1, keepdim=True)
    return kl


def _wasserstein_normal_normal(p, q):
    """
    Computes the squared Wasserstein-2 distance between two Normal distributions
    with diagonal covariance matrices.
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
    using the POT library.
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
    """Wasserstein distance between two policy distributions."""
    _, _, new_dist = new_actor.evaluate_actions(
        obs, rnn_states, action, masks, available_actions, active_masks
    )
    with torch.no_grad():
        _, _, old_dist = old_actor.evaluate_actions(
            obs, rnn_states, action, masks, available_actions, active_masks
        )
    if new_dist.__class__.__name__ == "FixedCategorical":
        w_dist = wasserstein_pot(old_dist.logits, new_dist.logits)
    else:
        w_dist = _wasserstein_normal_normal(old_dist, new_dist)
    if len(w_dist.shape) == 1:
        w_dist = w_dist.unsqueeze(1)
    return w_dist


def calculate_w_matrpo_loss(advantage, w_dist, lambda_val, delta):
    """
    Calculates actor and lambda losses for W-MATRPO based on its dual formulation.
    This corresponds to Equation (6) in the user's paper.

    Args:
        advantage (torch.Tensor): The advantage estimate M_i(s, a).
        w_dist (torch.Tensor): The computed Wasserstein distance.
        lambda_val (torch.Tensor): The current dual variable lambda.
        delta (float): The trust region radius delta.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - The loss for updating the policy network (actor_loss).
            - The loss for updating the dual variable (lambda_loss).
    """
    # Actor loss: Maximize advantage, penalized by the constraint violation.
    # Lambda is detached so its gradients don't affect the actor update.
    actor_loss = -advantage.mean() + lambda_val.detach() * (w_dist.mean() - delta)

    # Lambda loss: Minimize this to perform gradient ascent on lambda.
    # This increases lambda when the constraint is violated (w_dist > delta)
    # and decreases it otherwise.
    lambda_loss = -(lambda_val * (w_dist.mean() - delta))

    return actor_loss, lambda_loss
