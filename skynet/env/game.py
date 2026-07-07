"""Moteur de jeu Skyjo pur (règles), indépendant de toute API RL.

Grille : 3 lignes x 4 colonnes = 12 cellules, indexées 0..11 par
`row * 4 + col`. Une colonne `c` est formée des cellules [c, c+4, c+8].
"""

import random

from .cards import build_deck

HIDDEN = "hidden"
REVEALED = "revealed"
REMOVED = "removed"

N_CELLS = 12
N_COLS = 4
N_ROWS = 3


def column_cells(col):
    return [col, col + N_COLS, col + 2 * N_COLS]


class IllegalActionError(Exception):
    pass


class SkyjoGame:
    """Moteur de jeu à un seul round de Skyjo pour N joueurs (2-8).

    Usage: reset() puis step(action) répétés jusqu'à done=True.
    Les actions ne concernent que le joueur courant (`current_player`).
    """

    def __init__(self, n_players, rng=None):
        assert 2 <= n_players <= 8
        self.n_players = n_players
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        deck = build_deck()
        self.rng.shuffle(deck)
        self.deck = deck

        self.grids = []
        for _ in range(self.n_players):
            cells = []
            for _ in range(N_CELLS):
                cells.append({"value": self.deck.pop(), "state": HIDDEN})
            self.grids.append(cells)

        self.discard = [self.deck.pop()]

        self.phase = "initial_reveal"
        self.done = False
        self.final_round_trigger = None
        self.final_round_remaining = None
        self.scores = None
        self.turn_order = None
        self.current_turn_idx = 0
        self.last_action_log = []

        # Une pioche est un choix en deux temps : on tire la carte (elle
        # devient visible dans `pending_drawn_value`) puis, sachant sa
        # valeur, on décide de la placer ou de la défausser.
        self.turn_phase = "normal"  # "normal" | "pending_draw"
        self.pending_drawn_value = None

        self._run_initial_reveal()

    # ------------------------------------------------------------------
    # Mise en place
    # ------------------------------------------------------------------
    def _run_initial_reveal(self):
        for cells in self.grids:
            cells[0]["state"] = REVEALED

        candidates = list(range(self.n_players))
        next_cell_idx = 1
        while True:
            best_value = max(self.grids[p][0]["value"] for p in candidates) \
                if next_cell_idx == 1 else max(
                    self.grids[p][next_cell_idx - 1]["value"] for p in candidates
                )
            tied = [
                p for p in candidates
                if self.grids[p][next_cell_idx - 1]["value"] == best_value
            ]
            if len(tied) == 1 or next_cell_idx >= N_CELLS:
                starter = tied[0]
                break
            for p in tied:
                self.grids[p][next_cell_idx]["state"] = REVEALED
            candidates = tied
            next_cell_idx += 1

        self.turn_order = [
            (starter + i) % self.n_players for i in range(self.n_players)
        ]
        self.current_turn_idx = 0
        self.phase = "playing"

    # ------------------------------------------------------------------
    # Accesseurs
    # ------------------------------------------------------------------
    @property
    def current_player(self):
        return self.turn_order[self.current_turn_idx]

    def hidden_cells(self, player):
        return [i for i, c in enumerate(self.grids[player]) if c["state"] == HIDDEN]

    def active_cells(self, player):
        """Cellules non supprimées (hidden ou revealed)."""
        return [i for i, c in enumerate(self.grids[player]) if c["state"] != REMOVED]

    def legal_actions(self):
        """Retourne l'ensemble des actions légales pour current_player.

        Encodage (49 actions) :
          0..11  : reveal_hidden(cell)                     [phase normale]
          12..23 : place_drawn_card(cell)                  [après une pioche]
          24..35 : discard_drawn_reveal(cell)  # cell hidden [après une pioche]
          36..47 : take_discard_place(cell)                [phase normale]
          48     : draw_from_deck                          [phase normale]

        La pioche est un choix en deux temps : `draw_from_deck` tire la carte
        (visible ensuite via `pending_drawn_value`) sans terminer le tour ;
        le joueur choisit alors, en connaissance de la valeur, entre
        `place_drawn_card` et `discard_drawn_reveal`.
        """
        p = self.current_player
        hidden = set(self.hidden_cells(p))
        active = set(self.active_cells(p))
        legal = set()

        if self.turn_phase == "pending_draw":
            for i in active:
                legal.add(12 + i)
            for i in hidden:
                legal.add(24 + i)
            return legal

        for i in hidden:
            legal.add(i)  # reveal_hidden

        for i in active:
            legal.add(36 + i)  # take_discard_place

        can_draw_deck = len(self.deck) > 0 or len(self.discard) > 1
        if can_draw_deck:
            legal.add(48)

        return legal

    # ------------------------------------------------------------------
    # Déroulement d'un tour
    # ------------------------------------------------------------------
    def step(self, action):
        if self.done:
            raise IllegalActionError("La partie est terminée.")
        if action not in self.legal_actions():
            raise IllegalActionError(f"Action {action} illégale.")

        p = self.current_player
        triggered_end = False
        turn_over = True

        if self.turn_phase == "pending_draw":
            if 12 <= action < 24:
                cell = action - 12
                old = self.grids[p][cell]["value"]
                self.grids[p][cell] = {"value": self.pending_drawn_value, "state": REVEALED}
                self.discard.append(old)
                self._check_column(p, cell % N_COLS)
                triggered_end = self._check_hand_complete(p)
            elif 24 <= action < 36:
                cell = action - 24
                self.discard.append(self.pending_drawn_value)
                self._reveal(p, cell)
                triggered_end = self._check_hand_complete(p)
            else:
                raise IllegalActionError(f"Action {action} invalide en phase pending_draw.")
            self.turn_phase = "normal"
            self.pending_drawn_value = None

        elif 0 <= action < 12:
            cell = action
            self._reveal(p, cell)
            triggered_end = self._check_hand_complete(p)

        elif 36 <= action < 48:
            cell = action - 36
            drawn = self.discard.pop()
            old = self.grids[p][cell]["value"]
            self.grids[p][cell] = {"value": drawn, "state": REVEALED}
            self.discard.append(old)
            self._check_column(p, cell % N_COLS)
            triggered_end = self._check_hand_complete(p)

        elif action == 48:
            self._reshuffle_if_needed(1)
            self.pending_drawn_value = self.deck.pop()
            self.turn_phase = "pending_draw"
            turn_over = False

        else:
            raise IllegalActionError(f"Action {action} hors bornes.")

        if not turn_over:
            return self.done

        if triggered_end and self.phase == "playing":
            self.phase = "final_round"
            self.final_round_trigger = p
            self.final_round_remaining = self.n_players - 1

        self._advance_turn()
        return self.done

    def _reveal(self, player, cell):
        if self.grids[player][cell]["state"] != HIDDEN:
            raise IllegalActionError("Cellule déjà révélée ou supprimée.")
        self.grids[player][cell]["state"] = REVEALED
        self._check_column(player, cell % N_COLS)

    def _check_column(self, player, col):
        cells = column_cells(col)
        grid = self.grids[player]
        if all(grid[i]["state"] == REVEALED for i in cells):
            values = {grid[i]["value"] for i in cells}
            if len(values) == 1:
                for i in cells:
                    grid[i]["state"] = REMOVED

    def _check_hand_complete(self, player):
        return len(self.hidden_cells(player)) == 0

    def peek_deck(self):
        """Valeur de la prochaine carte de pioche, sans la retirer (déclenche le
        reshuffle si nécessaire, comme le fera le step qui suivra)."""
        self._reshuffle_if_needed(1)
        return self.deck[-1] if self.deck else None

    def _reshuffle_if_needed(self, needed):
        if len(self.deck) >= needed:
            return
        if len(self.discard) <= 1:
            return
        top = self.discard.pop()
        self.rng.shuffle(self.discard)
        self.deck = self.discard
        self.discard = [top]

    def _advance_turn(self):
        if self.phase == "final_round":
            self.final_round_remaining -= 1
            if self.final_round_remaining <= 0:
                self._end_game()
                return
        self.current_turn_idx = (self.current_turn_idx + 1) % self.n_players

    def _end_game(self):
        for cells in self.grids:
            for c in cells:
                if c["state"] == HIDDEN:
                    c["state"] = REVEALED
        self.scores = [self._score(p) for p in range(self.n_players)]
        self.done = True
        self.phase = "done"

    def _score(self, player):
        return sum(
            c["value"] for c in self.grids[player] if c["state"] == REVEALED
        )
