"""Serveur local Flask : interface web pour affronter Skynet."""

import json
import os
import random

import torch
from flask import Flask, jsonify, request, send_from_directory

from skynet.agents.expectimax import choose_action as choose_action_expectimax
from skynet.agents.network import ActorCriticNet
from skynet.env.game import SkyjoGame, HIDDEN, REMOVED
from skynet.env.skyjo_env import OBS_DIM, N_ACTIONS, observe, legal_action_mask
from skynet.env import legacy_v2_obs

DEVICE = torch.device("cpu")
CHECKPOINTS_ROOT = os.environ.get("SKYNET_CHECKPOINTS_ROOT", "checkpoints")
SLIDER_CONFIG_PATH = os.path.join(CHECKPOINTS_ROOT, "levels", "slider_config.json")

TIERS = ["beginner", "intermediate", "advanced"]
TIER_LABELS = {
    "beginner": "Débutant",
    "intermediate": "Intermédiaire",
    "advanced": "Avancé",
}

# Les 9 checkpoints entraînés, répartis par palier selon leur NIVEAU RÉEL
# (numéro de version ci-dessous, basé sur le score obtenu vs un adversaire
# aléatoire — un plus gros écart entre deux checkpoints d'une même lignée
# donne un plus gros saut de version), pas selon la lignée d'origine.
# Bornes utilisées : <2.0 Débutant, 2.0-3.5 Intermédiaire, >=3.5 Avancé.
# Benchmarks : 150 parties vs adversaire aléatoire, seed=42 (cf.
# checkpoints/levels/README.md pour le détail des runs d'entraînement).
def _variant(vid, dir_, file_, label, stats, lineage):
    is_legacy = lineage == "genesis"
    return {
        "id": vid, "dir": dir_, "file": file_, "label": label, "stats": stats,
        "obs_dim": legacy_v2_obs.OBS_DIM if is_legacy else OBS_DIM,
        "observe": legacy_v2_obs.observe if is_legacy else observe,
    }


VARIANTS = {
    "beginner": [
        _variant("genesis_10", "levels", "level_1.pt", "Genesis 1.0",
                  {"win_rate": 0.573, "avg_score": 44.77, "avg_placement": 0.55}, "genesis"),
        _variant("horizon_10", "levels_v3", "level_1.pt", "Horizon 1.0",
                  {"win_rate": 0.687, "avg_score": 43.53, "avg_placement": 0.46}, "horizon"),
        _variant("genesis_18", "levels", "level_2.pt", "Genesis 1.8",
                  {"win_rate": 0.867, "avg_score": 35.47, "avg_placement": 0.17}, "genesis"),
    ],
    "intermediate": [
        _variant("horizon_21", "levels_v3", "level_2.pt", "Horizon 2.1",
                  {"win_rate": 0.907, "avg_score": 32.90, "avg_placement": 0.10}, "horizon"),
        _variant("horizon_32", "levels_v3", "level_3.pt", "Horizon 3.2",
                  {"win_rate": 0.973, "avg_score": 24.30, "avg_placement": 0.03}, "horizon"),
        _variant("genesis_33", "levels", "level_3.pt", "Genesis 3.3",
                  {"win_rate": 0.980, "avg_score": 21.39, "avg_placement": 0.02}, "genesis"),
    ],
    "advanced": [
        _variant("genesis_40", "levels", "level_4.pt", "Genesis 4.0",
                  {"win_rate": 0.980, "avg_score": 17.75, "avg_placement": 0.02}, "genesis"),
        _variant("horizon_40", "levels_v3", "level_4.pt", "Horizon 4.0",
                  {"win_rate": 0.987, "avg_score": 19.89, "avg_placement": 0.01}, "horizon"),
        _variant("singularite_43", "levels_v4", "level_4.pt", "Singularité 4.3",
                  {"win_rate": 0.993, "avg_score": 18.49, "avg_placement": 0.01}, "singularite"),
    ],
}

DEFAULT_SLIDER_CONFIG = {
    "beginner": ["genesis_10", "horizon_10", "genesis_18"],
    "intermediate": ["horizon_21", "horizon_32", "genesis_33"],
    "advanced": ["genesis_40", "horizon_40", "singularite_43"],
    "expert_plus": True,
}

app = Flask(__name__, static_folder="static", static_url_path="")

STATE = {
    "game": None,
    "rng": random.Random(),
    "log": [],
    "seats_human": [],
    "level": 4,
}

_level_cache = {}  # level -> (net, mtime)


def load_slider_config():
    if os.path.exists(SLIDER_CONFIG_PATH):
        try:
            with open(SLIDER_CONFIG_PATH) as f:
                cfg = json.load(f)
            out = {}
            for tier in TIERS:
                valid_ids = {v["id"] for v in VARIANTS[tier]}
                ordered = [i for i in cfg.get(tier, []) if i in valid_ids]
                out[tier] = ordered or list(DEFAULT_SLIDER_CONFIG[tier])
            out["expert_plus"] = bool(cfg.get("expert_plus", DEFAULT_SLIDER_CONFIG["expert_plus"]))
            return out
        except (json.JSONDecodeError, OSError):
            pass
    return {
        **{tier: list(ids) for tier, ids in DEFAULT_SLIDER_CONFIG.items() if tier in TIERS},
        "expert_plus": DEFAULT_SLIDER_CONFIG["expert_plus"],
    }


def save_slider_config(cfg):
    os.makedirs(os.path.dirname(SLIDER_CONFIG_PATH), exist_ok=True)
    with open(SLIDER_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def build_level_meta(slider_config):
    meta = [{"level": 0, "label": "Aléatoire", "path": None, "obs_dim": None, "observe": None, "expectimax": False}]
    last_top_tier_variant = None
    top_tier = TIERS[-1]  # "advanced" : le palier le plus élevé, base d'Expert+
    for tier in TIERS:
        enabled = slider_config.get(tier, [])
        show_variant_name = len(enabled) > 1
        for vid in enabled:
            variant = next((v for v in VARIANTS[tier] if v["id"] == vid), None)
            if variant is None:
                continue
            path = os.path.join(CHECKPOINTS_ROOT, variant["dir"], variant["file"])
            label = f"{TIER_LABELS[tier]} ({variant['label']})" if show_variant_name else TIER_LABELS[tier]
            meta.append({
                "level": len(meta), "label": label, "path": path,
                "obs_dim": variant["obs_dim"], "observe": variant["observe"],
                "expectimax": False,
            })
            if tier == top_tier:
                last_top_tier_variant = variant
    # expectimax.py importe directement l'observation courante (skyjo_env.observe) ;
    # Expert+ n'a donc de sens que pour une variante qui partage cette même
    # architecture d'observation (obs_dim == OBS_DIM courant).
    if (
        slider_config.get("expert_plus", True)
        and last_top_tier_variant is not None
        and last_top_tier_variant["obs_dim"] == OBS_DIM
    ):
        path = os.path.join(CHECKPOINTS_ROOT, last_top_tier_variant["dir"], last_top_tier_variant["file"])
        meta.append({
            "level": len(meta), "label": "Expert+ (calcul)", "path": path,
            "obs_dim": last_top_tier_variant["obs_dim"], "observe": last_top_tier_variant["observe"],
            "expectimax": True,
        })
    return meta


SLIDER_CONFIG = load_slider_config()
LEVEL_META = build_level_meta(SLIDER_CONFIG)


def available_levels():
    out = []
    for meta in LEVEL_META:
        if meta["path"] is None or os.path.exists(meta["path"]):
            out.append({"level": meta["level"], "label": meta["label"]})
    return out


def get_policy_for_level(level):
    if level < 0 or level >= len(LEVEL_META):
        return None
    meta = LEVEL_META[level]
    if meta["path"] is None:
        return None
    path = meta["path"]
    mtime = os.path.getmtime(path)
    cache_key = (level, path)
    cached = _level_cache.get(cache_key)
    if cached is None or cached[1] != mtime:
        net = ActorCriticNet(meta["obs_dim"], N_ACTIONS).to(DEVICE)
        net.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
        net.eval()
        _level_cache[cache_key] = (net, mtime)
        print(f"Niveau {level} (re)chargé depuis {path}")
    return _level_cache[cache_key][0]


def ai_action(game, player):
    legal = sorted(game.legal_actions())
    level = STATE["level"]
    net = get_policy_for_level(level)

    if net is None:
        return STATE["rng"].choice(legal)

    meta = LEVEL_META[level]
    if meta.get("expectimax"):
        return choose_action_expectimax(game, player, net, device=DEVICE)

    obs = meta["observe"](game, player)
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


@app.route("/api/slider_options")
def slider_options():
    variants_with_availability = {}
    for tier in TIERS:
        items = []
        for v in VARIANTS[tier]:
            path = os.path.join(CHECKPOINTS_ROOT, v["dir"], v["file"])
            items.append({
                "id": v["id"], "label": v["label"], "stats": v["stats"],
                "available": os.path.exists(path),
            })
        variants_with_availability[tier] = items
    return jsonify({
        "tiers": TIERS,
        "tier_labels": TIER_LABELS,
        "variants": variants_with_availability,
        "config": SLIDER_CONFIG,
    })


@app.route("/api/slider_config", methods=["POST"])
def set_slider_config():
    global SLIDER_CONFIG, LEVEL_META
    data = request.get_json(force=True) or {}
    new_cfg = {}
    for tier in TIERS:
        valid_ids = {v["id"] for v in VARIANTS[tier]}
        requested = data.get(tier, [])
        if not isinstance(requested, list):
            requested = []
        ordered = [i for i in requested if i in valid_ids]
        new_cfg[tier] = ordered or list(DEFAULT_SLIDER_CONFIG[tier])
    new_cfg["expert_plus"] = bool(data.get("expert_plus", True))
    SLIDER_CONFIG = new_cfg
    save_slider_config(SLIDER_CONFIG)
    LEVEL_META = build_level_meta(SLIDER_CONFIG)
    _level_cache.clear()
    return jsonify({"ok": True, "levels": available_levels()})


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

    meta = LEVEL_META[level]
    if net is None:
        action = random.choice(legal)
    elif meta.get("expectimax"):
        action = choose_action_expectimax(game, 0, net, device=DEVICE)
    else:
        obs = meta["observe"](game, 0)
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
