# Niveaux Skynet — origine des checkpoints

> **⚠️ Le code de `skynet/env/skyjo_env.py` a été modifié (ajout de
> probabilités de complétion de colonne + score estimé par joueur,
> `OBS_DIM` passé de 1689 à 1701) mais AUCUN réentraînement n'a encore
> été lancé.** Tous les fichiers `.pt` ci-dessous, et le serveur
> actuellement en cours d'exécution, correspondent encore à
> l'ancienne architecture (`OBS_DIM = 1689`). Ne pas relancer/recharger
> le serveur tant qu'un réentraînement n'a pas produit de nouveaux
> checkpoints compatibles avec `OBS_DIM = 1701`, sous peine de crash au
> chargement (`load_state_dict`).

Architecture courante (v2) : pioche en deux temps (tirer puis décider
en connaissant la valeur) + comptage de cartes dans l'observation,
+ probabilités de complétion de colonne et score estimé par joueur.
`OBS_DIM = 1701`, `N_ACTIONS = 49` (voir `skynet/env/skyjo_env.py`).
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
| `level_4.pt` (même fichier) | Expert+ (calcul) | Mêmes poids qu'Expert — pas de réseau ni d'entraînement séparés. La différence est la procédure de décision : `skynet/agents/expectimax.py` évalue chaque issue incertaine (retourner/piocher/défausser) par une espérance pondérée sur les vraies probabilités de tirage (comptage de cartes), en utilisant la tête de valeur déjà entraînée, plutôt qu'un simple argmax sur les logits. |

`../skynet.pt` est le checkpoint "courant" qui s'écrase à chaque
sauvegarde périodique pendant l'entraînement (`--checkpoint-every`) ;
il sert à reprendre un entraînement (`--resume`, activé par défaut) et
correspond toujours à l'état le plus récent de la lignée Expert.

Pour ajouter un niveau : relancer `train.py` avec `--milestones` pointant
sur les itérations voulues, ou copier `checkpoints/skynet.pt` vers le
fichier de niveau cible une fois l'entraînement terminé — et mettre à
jour ce tableau.
