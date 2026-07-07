import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skynet.env.game import SkyjoGame, N_CELLS


def run_random_episode(n_players, seed):
    rng = random.Random(seed)
    game = SkyjoGame(n_players, rng=rng)
    steps = 0
    max_steps = 5000
    while not game.done and steps < max_steps:
        legal = list(game.legal_actions())
        assert legal, "Aucune action légale disponible alors que la partie n'est pas terminée."
        action = rng.choice(legal)
        game.step(action)
        steps += 1

    assert game.done, f"La partie ne s'est pas terminée en {max_steps} coups (n_players={n_players})."
    assert len(game.scores) == n_players

    for cells in game.grids:
        removed = [c for c in cells if c["state"] == "removed"]
        assert len(removed) % 3 == 0, "Les colonnes supprimées doivent l'être par groupes de 3."
        for col in range(4):
            col_states = {cells[col]["state"], cells[col + 4]["state"], cells[col + 8]["state"]}
            if "removed" in col_states:
                assert col_states == {"removed"}, "Colonne partiellement supprimée."

    return game.scores, steps


def main():
    total_deck_check = 0
    for n_players in range(2, 9):
        for seed in range(20):
            scores, steps = run_random_episode(n_players, seed)
            total_deck_check += 1
    print(f"OK: {total_deck_check} parties aléatoires terminées sans erreur (2 à 8 joueurs).")

    # Vérifie une partie précise pour inspection manuelle
    rng = random.Random(0)
    game = SkyjoGame(3, rng=rng)
    print("Ordre de tour initial:", game.turn_order)
    print("Scores premières manches (n=3, seed=0):",
          run_random_episode(3, 0)[0])


if __name__ == "__main__":
    main()
