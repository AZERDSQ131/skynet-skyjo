"""Boucle de mise à jour PPO (clipped surrogate) sur les échantillons collectés."""

import numpy as np
import torch
import torch.nn as nn

from .rollout import samples_to_tensors

MAX_PLAYERS = 8


def ppo_update(
    policy,
    optimizer,
    samples,
    device,
    clip_eps=0.2,
    epochs=4,
    batch_size=256,
    vf_coef=0.5,
    ent_coef=0.01,
    rank_coef=0.1,
    max_grad_norm=0.5,
):
    obs, mask, action, old_logprob, advantage, ret, rank, n_players = samples_to_tensors(samples, device)

    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    n = obs.shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "rank_loss": 0.0, "n_updates": 0}

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            b_obs, b_mask, b_action = obs[idx], mask[idx], action[idx]
            b_old_logprob, b_adv, b_ret = old_logprob[idx], advantage[idx], ret[idx]
            b_rank, b_n_players = rank[idx], n_players[idx]

            log_prob, entropy, value, rank_logits = policy.evaluate(b_obs, b_mask, b_action)
            ratio = torch.exp(log_prob - b_old_logprob)

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(value, b_ret)
            entropy_loss = entropy.mean()

            # Les classes >= n_players sont impossibles pour cette partie
            # (ex : rang 5 n'existe pas dans une partie à 3 joueurs).
            class_idx = torch.arange(MAX_PLAYERS, device=device).unsqueeze(0)
            valid = class_idx < b_n_players.unsqueeze(1)
            masked_rank_logits = rank_logits.masked_fill(~valid, -1e9)
            rank_loss = nn.functional.cross_entropy(masked_rank_logits, b_rank)

            loss = (
                policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
                + rank_coef * rank_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy_loss.item()
            stats["rank_loss"] += rank_loss.item()
            stats["n_updates"] += 1

    for k in ("policy_loss", "value_loss", "entropy", "rank_loss"):
        stats[k] /= max(stats["n_updates"], 1)
    return stats
