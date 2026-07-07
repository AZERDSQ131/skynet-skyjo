# Skynet — une IA Skyjo entraînée par self-play

Skynet apprend à jouer au [Skyjo](https://fr.wikipedia.org/wiki/Skyjo) depuis
zéro, par renforcement (PPO en self-play), et propose une interface web pour
l'affronter — ou pour se faire assister pendant une vraie partie physique.

## Le jeu

- Moteur de règles pur en Python (`skynet/env/game.py`), indépendant de
  toute API RL : grille 3×4, révélation initiale, colonnes identiques
  supprimées, dernier tour déclenché quand un joueur termine sa grille.
- La pioche est un choix **en deux temps** : on tire la carte, on voit sa
  valeur, puis seulement on décide de la placer ou de la défausser. C'est
  une correction volontaire par rapport à une première version où l'IA
  devait s'engager avant de voir la carte piochée — un bug qui la rendait
  incapable de saisir des opportunités évidentes (compléter une colonne
  avec la carte piochée, par exemple).

## L'IA

- **Self-play PPO** : un réseau acteur-critique (`skynet/agents/network.py`)
  partagé entre tous les joueurs, entraîné par PPO à politique clippée
  (`skynet/agents/ppo.py`) avec avantage GAE (`skynet/agents/rollout.py`).
- **Comptage de cartes** : l'observation inclut la fraction de chaque
  valeur encore inconnue (ni révélée, ni défaussée), plus des features
  dérivées explicites — probabilité qu'une carte inconnue complète une
  colonne, score estimé de chaque joueur (`skynet/env/skyjo_env.py`).
- **Décision expectimax à l'inférence** (niveau *Expert+*) : sans
  entraînement supplémentaire, une couche de décision évalue chaque issue
  incertaine (piocher, retourner, défausser) par une espérance pondérée
  sur les vraies probabilités de tirage, plutôt qu'un simple argmax
  (`skynet/agents/expectimax.py`).
- Plusieurs niveaux de difficulté sont sauvegardés à différentes étapes
  de l'entraînement (Débutant → Expert) et sélectionnables dans
  l'interface. Voir [`checkpoints/levels/README.md`](checkpoints/levels/README.md)
  pour l'origine exacte de chaque checkpoint.

## L'interface

Serveur Flask (`server.py`) + page unique (`static/index.html`), deux modes :

- **Jouer** : une vraie partie contre Skynet, plateau tournant, uniquement
  au clic.
- **Triche** : un mode assistant pour une partie physique — on recopie
  l'état des grilles (siennes et celles des adversaires) et Skynet indique
  le meilleur coup à jouer, en tenant compte du comptage de cartes.

## Lancer le projet

```bash
pip install -r requirements.txt
python3 server.py
# puis ouvrir http://127.0.0.1:5050
```

## Entraîner un nouveau modèle

```bash
python3 train.py --iterations 1400 --milestones "70,250,600,1400"
```

`train.py` sauvegarde périodiquement un checkpoint courant
(`checkpoints/skynet.pt`, reprise automatique avec `--resume`) et des
niveaux de difficulté figés à des itérations choisies
(`checkpoints/levels/level_*.pt`). Un changement de l'observation ou de
l'espace d'action rend les anciens checkpoints incompatibles (ils
échouent au chargement plutôt que de donner de mauvais résultats
silencieusement) — un réentraînement complet est alors nécessaire.

## Structure du projet

```
skynet/
  env/       moteur de jeu (game.py) + encodage RL (skyjo_env.py)
  agents/    réseau (network.py), PPO (ppo.py), rollout (rollout.py),
             décision expectimax (expectimax.py)
tests/       tests du moteur de jeu
server.py    backend Flask (API + logique de partie/triche)
static/      interface web (une seule page)
train.py     script d'entraînement self-play
```
