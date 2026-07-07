"""Serveur local Flask : interface web pour affronter Skynet."""

import os
import random

import torch
from flask import Flask, jsonify, request, send_from_directory

from skynet.agents.network import ActorCriticNet
from skynet.env.game import SkyjoGame, HIDDEN, REMOVED
from skynet.env.skyjo_env import OBS_DIM, N_ACTIONS, observe, legal_action_mask

DEVICE = torch.device("cpu")
LEVELS_DIR = os.environ.get("SKYNET_LEVELS_DIR", "checkpoints/levels")

LEVEL_META = [
    {"level": 0, "label": "Aléatoire", "file": None},
    {"level": 1, "label": "Débutant", "file": "level_1.pt"},
    {"level": 2, "label": "Intermédiaire", "file": "level_2.pt"},
    {"level": 3, "label": "Avancé", "file": "level_3.pt"},
    {"level": 4, "label": "Expert", "file": "level_4.pt"},
]

app = Flask(__name__, static_folder="static", static_url_path="")

STATE = {
    "game": None,
    "rng": random.Random(),
    "log": [],
    "seats_human": [],
    "level": 4,
}

_level_cache = {}  # level -> (net, mtime)


def available_levels():
    out = []
    for meta in LEVEL_META:
        if meta["file"] is None:
            out.append({"level": meta["level"], "label": meta["label"]})
            continue
        path = os.path.join(LEVELS_DIR, meta["file"])
        if os.path.exists(path):
            out.append({"level": meta["level"], "label": meta["label"]})
    return out


def get_policy_for_level(level):
    if level == 0:
        return None
    meta = LEVEL_META[level]
    path = os.path.join(LEVELS_DIR, meta["file"])
    mtime = os.path.getmtime(path)
    cached = _level_cache.get(level)
    if cached is None or cached[1] != mtime:
        net = ActorCriticNet(OBS_DIM, N_ACTIONS).to(DEVICE)
        net.load_state_dict(torch.load(path, map_location=DEVICE))
        net.eval()
        _level_cache[level] = (net, mtime)
        print(f"Niveau {level} (re)chargé depuis {path}")
    return _level_cache[level][0]


def ai_action(game, player):
    legal = sorted(game.legal_actions())
    net = get_policy_for_level(STATE["level"])

    if net is None:
        return STATE["rng"].choice(legal)

    obs = observe(game, player)
    mask = legal_action_mask(game)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits, _ = net.forward(obs_t)
        logits = logits.masked_fill(~mask_t, -1e9)
        action = int(torch.argmax(logits, dim=-1).item())
    return action


def describe_and_apply(game, action, label):
    p = game.current_player

    if game.turn_phase == "pending_draw":
        drawn = game.pending_drawn_value
        if 12 <= action < 24:
            cell = action - 12
            old = game.grids[p][cell]["value"]
            game.step(action)
            return f"{label} place la carte piochée ({drawn}) en case {cell} (défausse {old})"
        cell = action - 24
        game.step(action)
        revealed = game.grids[p][cell]["value"]
        return f"{label} défausse la carte piochée ({drawn}) et retourne la case {cell} : {revealed}"

    if 0 <= action < 12:
        cell = action
        value = game.grids[p][cell]["value"]
        game.step(action)
        return f"{label} retourne la case {cell} : {value}"

    if 36 <= action < 48:
        cell = action - 36
        taken = game.discard[-1]
        old = game.grids[p][cell]["value"]
        game.step(action)
        return f"{label} prend la défausse ({taken}) et la place en case {cell} (défausse {old})"

    game.step(action)  # action == 48 : pioche
    return f"{label} pioche : {game.pending_drawn_value}"


def describe_action_readonly(game, action):
    if game.turn_phase == "pending_draw":
        drawn = game.pending_drawn_value
        if 12 <= action < 24:
            cell = action - 12
            return {
                "action": "place_drawn", "cell": cell,
                "text": f"Place la carte piochée ({drawn}) ligne {cell // 4 + 1}, colonne {cell % 4 + 1}.",
            }
        cell = action - 24
        return {
            "action": "discard_drawn_reveal", "cell": cell,
            "text": f"Défausse la carte piochée ({drawn}) et retourne ta carte "
                    f"ligne {cell // 4 + 1}, colonne {cell % 4 + 1}.",
        }

    if 0 <= action < 12:
        cell = action
        return {
            "action": "reveal", "cell": cell,
            "text": f"Retourne ta carte ligne {cell // 4 + 1}, colonne {cell % 4 + 1}.",
        }
    if action == 48:
        return {
            "action": "draw", "cell": None,
            "text": "Pioche une carte, regarde sa valeur, puis redemande conseil pour savoir "
                    "si tu dois la placer ou la défausser.",
        }
    cell = action - 36
    taken = game.discard[-1]
    return {
        "action": "take_discard", "cell": cell,
        "text": f"Prends la carte de la défausse ({taken}) et place-la ligne {cell // 4 + 1}, "
                f"colonne {cell % 4 + 1}.",
    }


def is_human_seat(seat):
    seats_human = STATE["seats_human"]
    return seat < len(seats_human) and seats_human[seat]


def run_ai_turns():
    game = STATE["game"]
    while not game.done and not is_human_seat(game.current_player):
        p = game.current_player
        action = ai_action(game, p)
        desc = describe_and_apply(game, action, f"IA (siège {p + 1})")
        STATE["log"].append(desc)


def serialize_grid(cells, reveal_all=False):
    out = []
    for c in cells:
        if c["state"] == HIDDEN and not reveal_all:
            out.append({"state": "hidden", "value": None})
        elif c["state"] == REMOVED:
            out.append({"state": "removed", "value": None})
        else:
            out.append({"state": "revealed", "value": c["value"]})
    return out


def serialize_state():
    game = STATE["game"]
    players = []
    for p in range(game.n_players):
        players.append(
            {
                "id": p,
                "is_human": is_human_seat(p),
                "grid": serialize_grid(game.grids[p], reveal_all=game.done),
            }
        )

    legal = []
    if not game.done and is_human_seat(game.current_player):
        legal = sorted(game.legal_actions())

    return {
        "n_players": game.n_players,
        "current_player": game.current_player if not game.done else None,
        "level": STATE["level"],
        "phase": game.phase,
        "turn_phase": game.turn_phase,
        "pending_drawn_value": game.pending_drawn_value,
        "done": game.done,
        "scores": game.scores,
        "discard_top": game.discard[-1] if game.discard else None,
        "deck_count": len(game.deck),
        "final_round_remaining": game.final_round_remaining,
        "players": players,
        "legal_actions": legal,
        "log": STATE["log"][-40:],
    }


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/levels")
def levels():
    return jsonify(available_levels())


@app.route("/api/new_game", methods=["POST"])
def new_game():
    data = request.get_json(force=True) or {}
    n_players = int(data.get("n_players", 4))
    n_players = max(2, min(8, n_players))

    seats_human = data.get("seats_human")
    if not isinstance(seats_human, list) or len(seats_human) != n_players:
        seats_human = [i == 0 for i in range(n_players)]
    STATE["seats_human"] = [bool(x) for x in seats_human]

    valid_levels = {m["level"] for m in available_levels()}
    level = data.get("level", max(valid_levels) if valid_levels else 0)
    STATE["level"] = level if level in valid_levels else 0

    STATE["game"] = SkyjoGame(n_players, rng=STATE["rng"])
    STATE["log"] = []
    run_ai_turns()
    return jsonify(serialize_state())


@app.route("/api/state")
def state():
    if STATE["game"] is None:
        return jsonify({"error": "no_game"}), 400
    return jsonify(serialize_state())


@app.route("/api/action", methods=["POST"])
def action():
    game = STATE["game"]
    if game is None or game.done:
        return jsonify({"error": "no_active_game"}), 400
    if not is_human_seat(game.current_player):
        return jsonify({"error": "not_your_turn"}), 400

    data = request.get_json(force=True) or {}
    kind = data.get("kind")
    cell = data.get("cell")

    if kind == "draw":
        a = 48
    else:
        kind_to_offset = {
            "reveal": 0,
            "place_drawn": 12,
            "discard_drawn_reveal": 24,
            "take_discard": 36,
        }
        if kind not in kind_to_offset or cell is None:
            return jsonify({"error": "bad_request"}), 400
        a = kind_to_offset[kind] + int(cell)

    if a not in game.legal_actions():
        return jsonify({"error": "illegal_action"}), 400

    label = f"Joueur {game.current_player + 1}"
    desc = describe_and_apply(game, a, label)
    STATE["log"].append(desc)
    run_ai_turns()
    return jsonify(serialize_state())


def _parse_cheat_cells(cells):
    out = []
    for c in cells or []:
        state = c.get("state", "hidden")
        if state not in ("hidden", "revealed", "removed"):
            state = "hidden"
        value = c.get("value")
        out.append({"state": state, "value": int(value) if value is not None else 0})
    while len(out) < 12:
        out.append({"state": "hidden", "value": 0})
    return out[:12]


@app.route("/api/advise", methods=["POST"])
def advise():
    data = request.get_json(force=True) or {}
    my_grid = data.get("my_grid")
    discard_top = data.get("discard_top")
    if not my_grid or discard_top is None:
        return jsonify({"error": "invalid_state"}), 400

    opponents = data.get("opponents", [])
    deck_count = max(0, int(data.get("deck_count", 50) or 0))
    final_round = bool(data.get("final_round", False))

    game = SkyjoGame.__new__(SkyjoGame)
    game.rng = random.Random()
    game.grids = [_parse_cheat_cells(my_grid)] + [_parse_cheat_cells(o) for o in opponents]
    game.n_players = len(game.grids)
    game.discard = [int(discard_top)]
    game.deck = [0] * deck_count
    game.phase = "final_round" if final_round else "playing"
    if final_round:
        game.final_round_remaining = max(1, int(data.get("final_round_remaining", 1) or 1))
    else:
        game.final_round_remaining = None
    game.turn_order = list(range(game.n_players))
    game.current_turn_idx = 0
    game.done = False
    game.scores = None

    drawn_value = data.get("drawn_value")
    if drawn_value is not None:
        game.turn_phase = "pending_draw"
        game.pending_drawn_value = int(drawn_value)
    else:
        game.turn_phase = "normal"
        game.pending_drawn_value = None

    legal = sorted(game.legal_actions())
    if not legal:
        return jsonify({"error": "no_legal_actions"}), 400

    valid_levels = {m["level"] for m in available_levels()}
    level = data.get("level", 0)
    level = level if level in valid_levels else 0
    net = get_policy_for_level(level)

    if net is None:
        action = random.choice(legal)
    else:
        obs = observe(game, 0)
        mask = legal_action_mask(game)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, _ = net.forward(obs_t)
            logits = logits.masked_fill(~mask_t, -1e9)
            action = int(torch.argmax(logits, dim=-1).item())

    return jsonify(describe_action_readonly(game, action))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
