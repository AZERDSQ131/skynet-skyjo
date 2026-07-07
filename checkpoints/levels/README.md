# Niveaux Skynet — origine des checkpoints

Architecture courante (v2) : pioche en deux temps (tirer puis décider
en connaissant la valeur) + comptage de cartes dans l'observation.
`OBS_DIM = 1689`, `N_ACTIONS = 49` (voir `skynet/env/skyjo_env.py`).
Un fichier `.pt` n'est chargeable que par un réseau construit avec ces
mêmes dimensions — un changement d'architecture rend les anciens
fichiers incompatibles (ils échouent au chargement plutôt que de
donner de mauvais résultats silencieusement).

| Fichier | Niveau (UI) | Origine |
|---|---|---|
| `level_1.pt` | Débutant | Run `train_v2`, itération 70/1400 |
| `level_2.pt` | Intermédiaire | Run `train_v2`, itération 250/1400 |
| `level_3.pt` | Avancé | Run `train_v2`, itération 600/1400 |
| `level_4.pt` | Expert | Run `train_v2`, itération 1400/1400, puis raffiné par un run `train_refine` (700 itérations supplémentaires, entropie basse, repris depuis ce même checkpoint) |

`../skynet.pt` est le checkpoint "courant" qui s'écrase à chaque
sauvegarde périodique pendant l'entraînement (`--checkpoint-every`) ;
il sert à reprendre un entraînement (`--resume`, activé par défaut) et
correspond toujours à l'état le plus récent de la lignée Expert.

Pour ajouter un niveau : relancer `train.py` avec `--milestones` pointant
sur les itérations voulues, ou copier `checkpoints/skynet.pt` vers le
fichier de niveau cible une fois l'entraînement terminé — et mettre à
jour ce tableau.
