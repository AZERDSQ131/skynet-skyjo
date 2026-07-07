"""Collecte d'épisodes self-play et calcul des avantages (GAE) par joueur.

Chaque joueur d'une partie est traité comme sa propre trajectoire de
décisions (une politique partagée par tous), avec récompense nulle sauf
au dernier coup joué, où la récompense terminale relative est attribuée.
"""

import numpy as np
import torch

from skynet.env.skyjo_env import SkyjoEnv


def collect_episode(policy, device, gamma=0.99, lam=0.95, rng=None):
    env = SkyjoEnv(rng=rng)
    trajectories = {p: {"obs": [], "mask": [], "action": [], "logprob": [], "value": []}
                     for p in range(env.n_players)}

    while not env.done:
        player = env.current_player_id()
        obs_vec = env.observe(player)
        mask = env.legal_action_mask()

        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        action, logprob, value = policy.act(obs_t, mask_t)

        a = int(action.item())
        traj = trajectories[player]
        traj["obs"].append(obs_vec)
        traj["mask"].append(mask)
        traj["action"].append(a)
        traj["logprob"].append(float(logprob.item()))
        traj["value"].append(float(value.item()))

        env.step(a)

    rewards = env.terminal_rewards()
    sorted_scores = sorted(env.scores)
    ranks = {p: sorted_scores.index(env.scores[p]) for p in range(env.n_players)}

    samples = []
    for p, traj in trajectories.items():
        T = len(traj["obs"])
        if T == 0:
            continue
        rews = [0.0] * T
        rews[-1] = float(rewards[p])
        values = traj["value"] + [0.0]

        advantages = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            delta = rews[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        returns = [advantages[t] + values[t] for t in range(T)]

        for t in range(T):
            samples.append(
                {
                    "obs": traj["obs"][t],
                    "mask": traj["mask"][t],
                    "action": traj["action"][t],
                    "logprob": traj["logprob"][t],
                    "advantage": advantages[t],
                    "return": returns[t],
                    "rank": ranks[p],
                    "n_players": env.n_players,
                }
            )

    return samples, env.scores, env.n_players


def collect_rollout(policy, device, n_episodes, gamma=0.99, lam=0.95, rng=None):
    all_samples = []
    all_scores = []
    all_n_players = []
    for _ in range(n_episodes):
        samples, scores, n_players = collect_episode(policy, device, gamma, lam, rng)
        all_samples.extend(samples)
        all_scores.append(scores)
        all_n_players.append(n_players)
    return all_samples, all_scores, all_n_players


def samples_to_tensors(samples, device):
    obs = torch.as_tensor(np.array([s["obs"] for s in samples]), dtype=torch.float32, device=device)
    mask = torch.as_tensor(np.array([s["mask"] for s in samples]), dtype=torch.bool, device=device)
    action = torch.as_tensor([s["action"] for s in samples], dtype=torch.long, device=device)
    old_logprob = torch.as_tensor([s["logprob"] for s in samples], dtype=torch.float32, device=device)
    advantage = torch.as_tensor([s["advantage"] for s in samples], dtype=torch.float32, device=device)
    ret = torch.as_tensor([s["return"] for s in samples], dtype=torch.float32, device=device)
    rank = torch.as_tensor([s["rank"] for s in samples], dtype=torch.long, device=device)
    n_players = torch.as_tensor([s["n_players"] for s in samples], dtype=torch.long, device=device)
    return obs, mask, action, old_logprob, advantage, ret, rank, n_players
