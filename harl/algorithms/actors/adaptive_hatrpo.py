"""Adaptive HATRPO algorithm."""
""" WIP: Will need changes to incorporate different file imports """
import numpy as np
import torch
from copy import deepcopy
from harl.utils.envs_tools import check
from harl.utils.trpo_util import (
    flat_grad,
    flat_params,
    conjugate_gradient,
    fisher_vector_product,
    update_model,
    kl_divergence
)
from harl.utils.wasserstein_trpo_util import (
    wasserstein_distance_1d,
    calculate_adaptive_trust_region
)
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.models.policy_models.stochastic_policy import StochasticPolicy


class Adaptive_HATRPO(OnPolicyBase):
    """
    Adaptive HATRPO algorithm class.
    Inherits from HATRPO and adds the Coordination-Aware Adaptive Trust Region (CAATR) mechanism.
    """
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize Adaptive HATRPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        assert (
            act_space.__class__.__name__ != "MultiDiscrete"
        ), "only continuous and discrete action space is supported by Adaptive HATRPO."
        super(Adaptive_HATRPO, self).__init__(args, obs_space, act_space, device)   #namechange

        # Standard TRPO hyperparameters
        self.kl_threshold = args["kl_threshold"]
        self.ls_step = args["ls_step"]
        self.accept_ratio = args["accept_ratio"]
        self.backtrack_coeff = args["backtrack_coeff"]
        
        # New hyperparameters for the adaptive mechanism
        self.use_adaptive_kl = args.get("use_adaptive_kl", False)
        self.adaptive_C = args.get("adaptive_C", 0.1)
        self.adaptive_epsilon = args.get("adaptive_epsilon", 1e-8)
        
        # Storage for historical policies needed for drift calculation
        self.actor_t_minus_1 = deepcopy(self.actor)
        self.actor_t_minus_2 = deepcopy(self.actor)

    def update_historical_policies(self):
        """
        Updates the stored historical policies after a training step.
        The current actor becomes t-1, and t-1 becomes t-2.
        """
        self.actor_t_minus_2.load_state_dict(self.actor_t_minus_1.state_dict())
        self.actor_t_minus_1.load_state_dict(self.actor.state_dict())

    def _calculate_teammate_drift(self, actor_buffer, all_agent_historical_policies, current_agent_id):
        """
        Calculates the sum of policy drift (KL divergence) for all teammates.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data.
            all_agent_historical_policies (dict): Dict containing historical policies
                                                  of all agents.
            current_agent_id (int): The ID of the agent currently being updated.
        Returns:
            (float): The sum of KL divergences for all teammates.
        """
        # We need a sample batch to compute the expected KL divergence
        sample = next(actor_buffer.feed_forward_generator_actor(actor_buffer.advantages, 1))
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            _, _, _, _,
            available_actions_batch,
            _,
        ) = sample

        total_drift = 0.0
        num_teammates = 0

        for agent_id, historical_policies in all_agent_historical_policies.items():
            if agent_id == current_agent_id:
                continue

            policy_t_minus_1 = historical_policies['t-1']
            policy_t_minus_2 = historical_policies['t-2']

            # Calculate KL divergence between the teammate's last two policies
            kl = kl_divergence(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch=None, # Drift is calculated over all states
                new_actor=policy_t_minus_1,
                old_actor=policy_t_minus_2,
            )
            total_drift += kl.mean().item()
            num_teammates += 1

        return total_drift if num_teammates > 0 else 0.0

    def update(self, sample):
        """
        Update actor network. This is the original HATRPO update logic.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            kl: (torch.Tensor) KL divergence between old and new policy.
            loss_improve: (np.float32) loss improvement.
            expected_improve: (np.ndarray) expected loss improvement.
            dist_entropy: (torch.Tensor) action entropies.
            ratio: (torch.Tensor) ratio between new and old policy.
        """

        (
            obs_batch, rnn_states_batch, actions_batch, masks_batch,
            active_masks_batch, old_action_log_probs_batch, adv_targ,
            available_actions_batch, factor_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch, rnn_states_batch, actions_batch, masks_batch,
            available_actions_batch, active_masks_batch,
        )

        # actor update
        ratio = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1, keepdim=True,
        )
        if self.use_policy_active_masks:
            loss = (
                torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            loss = torch.sum(
                ratio * factor_batch * adv_targ, dim=-1, keepdim=True
            ).mean()

        loss_grad = torch.autograd.grad(loss, self.actor.parameters(), allow_unused=True)
        loss_grad = flat_grad(loss_grad)

        step_dir = conjugate_gradient(
            self.actor, obs_batch, rnn_states_batch, actions_batch,
            masks_batch, available_actions_batch, active_masks_batch,
            loss_grad.data, nsteps=10, device=self.device,
        )

        loss = loss.data.cpu().numpy()
        params = flat_params(self.actor)
        fvp = fisher_vector_product(
            self.actor, obs_batch, rnn_states_batch, actions_batch,
            masks_batch, available_actions_batch, active_masks_batch, step_dir
        )
        shs = 0.5 * (step_dir * fvp).sum(0, keepdim=True)
        step_size = 1 / torch.sqrt(shs / self.kl_threshold)[0]
        full_step = step_size * step_dir

        old_actor = StochasticPolicy(self.args, self.obs_space, self.act_space, self.device)
        update_model(old_actor, params)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
        expected_improve = expected_improve.data.cpu().numpy()

        # Backtracking line search (https://en.wikipedia.org/wiki/Backtracking_line_search)
        flag = False
        fraction = 1
        for i in range(self.ls_step):
            new_params = params + fraction * full_step
            update_model(self.actor, new_params)
            action_log_probs, dist_entropy, _ = self.evaluate_actions(
                obs_batch, rnn_states_batch, actions_batch, masks_batch,
                available_actions_batch, active_masks_batch,
            )

            ratio = getattr(torch, self.action_aggregation)(
                torch.exp(action_log_probs - old_action_log_probs_batch),
                dim=-1, keepdim=True,
            )
            if self.use_policy_active_masks:
                new_loss = (
                    torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True)
                    * active_masks_batch
                ).sum() / active_masks_batch.sum()
            else:
                new_loss = torch.sum(
                    ratio * factor_batch * adv_targ, dim=-1, keepdim=True
                ).mean()

            new_loss = new_loss.data.cpu().numpy()
            loss_improve = new_loss - loss

            kl = kl_divergence(
                obs_batch, rnn_states_batch, actions_batch, masks_batch,
                available_actions_batch, active_masks_batch,
                new_actor=self.actor, old_actor=old_actor,
            )
            kl = kl.mean()

            if (
                kl < self.kl_threshold
                and (loss_improve / expected_improve) > self.accept_ratio
                and loss_improve.item() > 0
            ):
                flag = True
                break
            expected_improve *= self.backtrack_coeff 
            #should above line be removed?
            fraction *= self.backtrack_coeff

        if not flag:
            params = flat_params(old_actor)
            update_model(self.actor, params)
            print("policy update does not impove the surrogate")

        return kl, loss_improve, expected_improve, dist_entropy, ratio

    def train(self, actor_buffer, advantages, state_type, all_agent_historical_policies=None, agent_id=None):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
            all_agent_historical_policies (dict): Dict containing historical policies of all agents.
            agent_id (int): The ID of the agent currently being updated.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["kl"] = 0
        train_info["dist_entropy"] = 0
        train_info["loss_improve"] = 0
        train_info["expected_improve"] = 0
        train_info["ratio"] = 0

        # Adaptive KL-threshold calculation
        if self.use_adaptive_kl and all_agent_historical_policies is not None and agent_id is not None:
            teammate_drift = self._calculate_teammate_drift(actor_buffer, all_agent_historical_policies, agent_id)
            adaptive_delta = self.adaptive_C / (teammate_drift + self.adaptive_epsilon)
            self.kl_threshold = adaptive_delta
            train_info["adaptive_kl_threshold"] = self.kl_threshold
            
        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self.use_recurrent_policy:
            data_generator = actor_buffer.recurrent_generator_actor(advantages, 1, self.data_chunk_length)
        elif self.use_naive_recurrent_policy:
            data_generator = actor_buffer.naive_recurrent_generator_actor(advantages, 1)
        else:
            data_generator = actor_buffer.feed_forward_generator_actor(advantages, 1)

        for sample in data_generator:
            kl, loss_improve, expected_improve, dist_entropy, imp_weights = self.update(sample)
            train_info["kl"] += kl
            train_info["loss_improve"] += loss_improve.item()
            train_info["expected_improve"] += expected_improve
            train_info["dist_entropy"] += dist_entropy.item()
            train_info["ratio"] += imp_weights.mean()

        num_updates = 1
        for k in train_info.keys():
            train_info[k] /= num_updates
            
        # After the update, save the new policy for the next iteration's calculation
        self.update_historical_policies()

        return train_info
