# Niveaux Skynet — origine des checkpoints

> **⚠️ Un run `train_v3` (nouvelle architecture, `OBS_DIM = 1701`) est
> terminé et disponible dans `checkpoints/levels_v3/` +
> `checkpoints/skynet_v3.pt`, mais n'est PAS encore utilisé par le
> serveur.** Le serveur actuellement en cours d'exécution charge
> toujours `checkpoints/levels/level_*.pt`, qui correspondent à
> l'ancienne architecture (`OBS_DIM = 1689`, run `train_v2`/`train_refine`
> ci-dessous). Les deux lignées sont volontairement gardées côte à côte
> dans des dossiers séparés le temps de comparer/valider — ne pas copier
> `levels_v3/*.pt` par-dessus `levels/*.pt` ni redémarrer le serveur sans
> décision explicite de l'utilisateur, sous peine de crash au chargement
> (`load_state_dict`) si jamais les deux architectures se mélangent.

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

## Lignée v3 (`OBS_DIM = 1701`, pas encore active)

Run `train_v3` : entraînement complet depuis zéro (`--no-resume`), mêmes
réglages que `train_v2` (1400 itérations, mêmes paliers d'entropie
0.02→0.002, mêmes milestones 70/250/600/1400), mais avec les deux
nouvelles features d'observation ajoutées à `skyjo_env.py` (probabilité
de complétion de colonne, score estimé par joueur). Éval finale vs
aléatoire : 100 % de victoires, score moyen 20.31.

| Fichier (`checkpoints/levels_v3/`) | Correspondrait à | Origine |
|---|---|---|
| `level_1.pt` | Débutant | Run `train_v3`, itération 70/1400 |
| `level_2.pt` | Intermédiaire | Run `train_v3`, itération 250/1400 |
| `level_3.pt` | Avancé | Run `train_v3`, itération 600/1400 |
| `level_4.pt` | Expert | Run `train_v3`, itération 1400/1400 |

`../skynet_v3.pt` est l'équivalent du checkpoint "courant" pour cette
lignée (utilisable avec `--resume` pour continuer/raffiner ce run
précis, incompatible avec `../skynet.pt` de la lignée v2).

Pour basculer le serveur sur cette lignée : remplacer les fichiers dans
`checkpoints/levels/` par ceux de `checkpoints/levels_v3/` (ou changer
`LEVELS_DIR` dans `server.py`), puis redémarrer le serveur. À faire
uniquement sur décision explicite de l'utilisateur, après comparaison.

Pour ajouter un niveau : relancer `train.py` avec `--milestones` pointant
sur les itérations voulues, ou copier `checkpoints/skynet.pt` vers le
fichier de niveau cible une fois l'entraînement terminé — et mettre à
jour ce tableau.
