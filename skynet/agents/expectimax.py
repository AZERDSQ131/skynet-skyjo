"""Amélioration de décision à l'inférence, sans entraînement supplémentaire.

Le réseau acteur-critique déjà entraîné choisit normalement une action
par un simple argmax sur ses logits, sans jamais évaluer explicitement
"qu'est-ce que je risque de tirer ?". Ce module ajoute une couche
expectimax à un coup : pour chaque action candidate dont l'issue est
incertaine (retourner une carte cachée, piocher, défausser puis
retourner), on énumère les valeurs encore possibles pondérées par leur
vraie probabilité (comptage de cartes), on évalue chaque résultat
hypothétique avec la tête de valeur déjà entraînée, et on choisit
l'action dont l'espérance de valeur est la meilleure.
"""

import random

import numpy as np
import torch

from skynet.env.game import SkyjoGame, REVEALED
from skynet.env.skyjo_env import observe, remaining_card_counts


def _clone_game(game):
    g = SkyjoGame.__new__(SkyjoGame)
    g.n_players = game.n_players
    g.grids = [[dict(cell) for cell in grid] for grid in game.grids]
    g.discard = list(game.discard)
    g.deck = list(game.deck)
    g.phase = game.phase
    g.done = False
    g.turn_order = list(game.turn_order)
    g.current_turn_idx = game.current_turn_idx
    g.final_round_remaining = game.final_round_remaining
    g.final_round_trigger = game.final_round_trigger
    g.turn_phase = game.turn_phase
    g.pending_drawn_value = game.pending_drawn_value
    g.scores = None
    g.rng = random.Random()
    return g


def _expected_value(game, player, net, device, mutate, uncertain):
    """mutate(clone, value_or_None) modifie clone en place pour une issue
    hypothétique donnée. Si uncertain, on moyenne sur les valeurs encore
    possibles (comptage de cartes) ; sinon l'issue est déterministe."""
    obs_batch = []
    weights = []

    if not uncertain:
        clone = _clone_game(game)
        mutate(clone, None)
        obs_batch.append(observe(clone, player))
        weights.append(1.0)
    else:
        counts = remaining_card_counts(game)
        total = sum(counts.values())
        if total <= 0:
            clone = _clone_game(game)
            mutate(clone, None)
            obs_batch.append(observe(clone, player))
            weights.append(1.0)
        else:
            for value, count in counts.items():
                if count <= 0:
                    continue
                clone = _clone_game(game)
                mutate(clone, value)
                obs_batch.append(observe(clone, player))
                weights.append(count / total)

    obs_t = torch.as_tensor(np.array(obs_batch), dtype=torch.float32, device=device)
    with torch.no_grad():
        _, values = net.forward(obs_t)
    values = values.tolist()
    return sum(w * v for w, v in zip(weights, values))


def _mutate_reveal(cell):
    def fn(clone, value):
        p = clone.current_player
        clone.grids[p][cell] = {"value": value, "state": REVEALED}
        clone._check_column(p, cell % 4)
    return fn


def _mutate_take_discard(cell):
    def fn(clone, _value):
        p = clone.current_player
        taken = clone.discard[-1]
        old = clone.grids[p][cell]["value"]
        clone.grids[p][cell] = {"value": taken, "state": REVEALED}
        clone.discard.append(old)
        clone._check_column(p, cell % 4)
    return fn


def _mutate_draw():
    def fn(clone, value):
        clone.turn_phase = "pending_draw"
        clone.pending_drawn_value = value
    return fn


def _mutate_place_drawn(cell):
    def fn(clone, _value):
        p = clone.current_player
        drawn = clone.pending_drawn_value
        old = clone.grids[p][cell]["value"]
        clone.grids[p][cell] = {"value": drawn, "state": REVEALED}
        clone.discard.append(old)
        clone._check_column(p, cell % 4)
        clone.turn_phase = "normal"
        clone.pending_drawn_value = None
    return fn


def _mutate_discard_drawn_reveal(cell):
    def fn(clone, value):
        p = clone.current_player
        clone.discard.append(clone.pending_drawn_value)
        clone.grids[p][cell] = {"value": value, "state": REVEALED}
        clone._check_column(p, cell % 4)
        clone.turn_phase = "normal"
        clone.pending_drawn_value = None
    return fn


def choose_action(game, player, net, device=None):
    """Choisit la meilleure action légale par expectimax à un coup, en
    s'appuyant sur la tête de valeur de `net` (déjà entraîné, inchangé)."""
    device = device or torch.device("cpu")
    legal = sorted(game.legal_actions())
    scored = []

    if game.turn_phase == "pending_draw":
        for a in legal:
            if 12 <= a < 24:
                cell = a - 12
                score = _expected_value(game, player, net, device, _mutate_place_drawn(cell), uncertain=False)
            else:
                cell = a - 24
                score = _expected_value(game, player, net, device, _mutate_discard_drawn_reveal(cell), uncertain=True)
            scored.append((score, a))
    else:
        for a in legal:
            if a == 48:
                score = _expected_value(game, player, net, device, _mutate_draw(), uncertain=True)
            elif 0 <= a < 12:
                score = _expected_value(game, player, net, device, _mutate_reveal(a), uncertain=True)
            else:
                cell = a - 36
                score = _expected_value(game, player, net, device, _mutate_take_discard(cell), uncertain=False)
            scored.append((score, a))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]
