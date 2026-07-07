"""Encodage d'observation de la lignée v2 (`OBS_DIM = 1689`), conservé tel
quel pour pouvoir continuer à servir les checkpoints `checkpoints/levels/`
même après l'ajout des features v3 (proba de complétion de colonne, score
estimé) qui ont changé `skynet.env.skyjo_env.OBS_DIM`. Ne pas modifier :
un changement ici casserait le chargement des anciens checkpoints v2.
"""

import numpy as np

from .cards import NUM_VALUES, value_to_index
from .game import N_CELLS, HIDDEN, REVEALED, REMOVED
from .skyjo_env import MAX_PLAYERS, N_OPP_SLOTS, _card_counts_remaining_frac

CELL_DIM = 2 + NUM_VALUES
OWN_GRID_DIM = N_CELLS * CELL_DIM
OPP_SLOT_DIM = 1 + OWN_GRID_DIM

OBS_DIM = (
    OWN_GRID_DIM
    + N_OPP_SLOTS * OPP_SLOT_DIM
    + NUM_VALUES
    + 1 + 1 + 1 + 1
    + NUM_VALUES
    + 1
    + NUM_VALUES
)


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
