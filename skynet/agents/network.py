"""Réseau acteur-critique partagé, utilisé par tous les joueurs (self-play)."""

import torch
import torch.nn as nn
from torch.distributions import Categorical

NEG_INF = -1e9


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def _masked_dist(self, logits, action_mask):
        masked_logits = logits.masked_fill(~action_mask, NEG_INF)
        return Categorical(logits=masked_logits)

    def act(self, obs, action_mask):
        """obs: (B, obs_dim) float32, action_mask: (B, action_dim) bool.

        Retourne action, log_prob, value (tous détachés, pour la collecte).
        """
        with torch.no_grad():
            logits, value = self.forward(obs)
            dist = self._masked_dist(logits, action_mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs, action_mask, action):
        """Utilisé pendant l'update PPO (avec gradient)."""
        logits, value = self.forward(obs)
        dist = self._masked_dist(logits, action_mask)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value
