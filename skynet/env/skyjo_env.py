"""Wrapper RL au-dessus du moteur SkyjoGame : encodage d'observation,
masque d'actions légales, et calcul de la récompense terminale.

Ce n'est pas un gymnasium.Env classique (un seul agent) mais un env
multi-agent séquentiel : à chaque instant, `current_player_id()` indique
qui doit jouer, et `observe(player)` / `legal_action_mask()` donnent tout
ce qu'il faut pour décider. La politique est partagée entre tous les
joueurs (self-play à paramètres partagés).
"""

import numpy as np

from .cards import CARD_COUNTS, NUM_VALUES, value_to_index
from .game import SkyjoGame, N_CELLS, HIDDEN, REVEALED, REMOVED

MAX_PLAYERS = 8
N_ACTIONS = 49

CELL_DIM = 2 + NUM_VALUES  # hidden_flag, removed_flag, onehot(valeur)
OWN_GRID_DIM = N_CELLS * CELL_DIM
OPP_SLOT_DIM = 1 + N_CELLS * CELL_DIM  # active_flag + grille
N_OPP_SLOTS = MAX_PLAYERS - 1

OBS_DIM = (
    OWN_GRID_DIM
    + N_OPP_SLOTS * OPP_SLOT_DIM
    + NUM_VALUES  # défausse (top, one-hot)
    + 1  # fraction de pioche restante
    + 1  # fraction de joueurs actifs
    + 1  # flag dernier tour
    + 1  # fraction de tours restants en dernier tour
    + NUM_VALUES  # carte piochée en attente (one-hot), 0 si aucune
    + 1  # flag "pioche en attente de décision"
    + NUM_VALUES  # comptage de cartes : fraction encore inconnue par valeur
)


def remaining_card_counts(game):
    """Nombre d'exemplaires de chaque valeur encore inconnus (ni révélés,
    ni défaussés, ni la carte piochée en attente) : dans la pioche ou
    dans une main cachée. Sert au comptage de cartes (probabilités
    réelles de tirage), pas seulement à l'observation du réseau."""
    seen = {v: 0 for v in CARD_COUNTS}
    for grid in game.grids:
        for cell in grid:
            if cell["state"] in (REVEALED, REMOVED):
                seen[cell["value"]] += 1
    for v in game.discard:
        seen[v] += 1
    if game.pending_drawn_value is not None:
        seen[game.pending_drawn_value] += 1
    return {v: max(0, total - seen[v]) for v, total in CARD_COUNTS.items()}


def _card_counts_remaining_frac(game):
    """Pour chaque valeur, fraction des exemplaires encore inconnus."""
    remaining = remaining_card_counts(game)
    frac = np.zeros(NUM_VALUES, dtype=np.float32)
    for v, total in CARD_COUNTS.items():
        frac[value_to_index(v)] = remaining[v] / total
    return frac


def _encode_cell(cell, hide_value):
    vec = np.zeros(CELL_DIM, dtype=np.float32)
    if cell["state"] == HIDDEN:
        vec[0] = 1.0
        return vec
    if cell["state"] == REMOVED:
        vec[1] = 1.0
        return vec
    if hide_value:
        return vec
    vec[2 + value_to_index(cell["value"])] = 1.0
    return vec


def _encode_grid(grid, hide_values):
    return np.concatenate([_encode_cell(c, hide_values) for c in grid])


def legal_action_mask(game):
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    for a in game.legal_actions():
        mask[a] = True
    return mask


def observe(game, player):
    n = game.n_players
    turn_order = game.turn_order
    current_idx = turn_order.index(player)

    own_vec = _encode_grid(game.grids[player], hide_values=False)

    opp_parts = []
    for slot in range(N_OPP_SLOTS):
        k = slot + 1
        if k < n:
            opp_idx = (current_idx + k) % n
            opp_player = turn_order[opp_idx]
            active_flag = np.array([1.0], dtype=np.float32)
            grid_vec = _encode_grid(game.grids[opp_player], hide_values=True)
        else:
            active_flag = np.array([0.0], dtype=np.float32)
            grid_vec = np.zeros(OWN_GRID_DIM, dtype=np.float32)
        opp_parts.append(np.concatenate([active_flag, grid_vec]))

    discard_vec = np.zeros(NUM_VALUES, dtype=np.float32)
    if game.discard:
        discard_vec[value_to_index(game.discard[-1])] = 1.0

    deck_frac = np.array([len(game.deck) / 150.0], dtype=np.float32)
    n_active_frac = np.array([n / MAX_PLAYERS], dtype=np.float32)
    final_round_flag = np.array(
        [1.0 if game.phase == "final_round" else 0.0], dtype=np.float32
    )
    if game.phase == "final_round" and n > 1:
        remaining_frac = np.array(
            [game.final_round_remaining / (n - 1)], dtype=np.float32
        )
    else:
        remaining_frac = np.array([0.0], dtype=np.float32)

    pending_vec = np.zeros(NUM_VALUES, dtype=np.float32)
    pending_flag = np.array(
        [1.0 if game.turn_phase == "pending_draw" else 0.0], dtype=np.float32
    )
    if game.turn_phase == "pending_draw" and game.pending_drawn_value is not None:
        pending_vec[value_to_index(game.pending_drawn_value)] = 1.0

    count_frac = _card_counts_remaining_frac(game)

    obs = np.concatenate(
        [own_vec, *opp_parts, discard_vec, deck_frac, n_active_frac,
         final_round_flag, remaining_frac, pending_vec, pending_flag, count_frac]
    )
    assert obs.shape[0] == OBS_DIM
    return obs


def terminal_rewards(game, scale=10.0):
    """Récompense relative par joueur : (score moyen des autres - son score) / scale."""
    scores = np.array(game.scores, dtype=np.float32)
    n = len(scores)
    total = scores.sum()
    rewards = np.zeros(n, dtype=np.float32)
    for i in range(n):
        others_mean = (total - scores[i]) / (n - 1)
        rewards[i] = (others_mean - scores[i]) / scale
    return rewards


class SkyjoEnv:
    """API multi-agent séquentielle autour de SkyjoGame."""

    def __init__(self, rng=None, n_players=None):
        self.rng = rng
        self._fixed_n_players = n_players
        self.game = None
        self.reset()

    def reset(self, n_players=None):
        n = n_players or self._fixed_n_players
        if n is None:
            import random

            n = (self.rng or random).randint(2, 8)
        self.game = SkyjoGame(n, rng=self.rng)
        return self.observe(self.game.current_player)

    @property
    def n_players(self):
        return self.game.n_players

    @property
    def done(self):
        return self.game.done

    @property
    def scores(self):
        return self.game.scores

    def current_player_id(self):
        return self.game.current_player

    def legal_action_mask(self):
        return legal_action_mask(self.game)

    def observe(self, player):
        return observe(self.game, player)

    def step(self, action):
        """Joue l'action pour current_player. Retourne (done, scores|None)."""
        done = self.game.step(action)
        if done:
            return True, self.game.scores
        return False, None

    def terminal_rewards(self, scale=10.0):
        return terminal_rewards(self.game, scale=scale)
