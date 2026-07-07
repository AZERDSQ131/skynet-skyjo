# Niveaux Skynet — origine des checkpoints

> **Le serveur sert maintenant plusieurs lignées à la fois**, via un
> panneau de réglages (bouton "⚙️ Réglages" dans l'interface) qui permet
> de choisir, palier par palier (Débutant/Intermédiaire/Avancé/Expert),
> quelle variante (v2/v3/v4) utiliser. La configuration est persistée
> dans `checkpoints/levels/slider_config.json`. Comme v2 (`OBS_DIM=1689`)
> et v3/v4 (`OBS_DIM=1701`) ne sont pas binairement compatibles,
> `server.py` construit un réseau avec la bonne dimension par variante
> (`skynet/env/legacy_v2_obs.py` reconstruit l'encodage v2 à l'identique
> pour cette raison — ne pas le modifier, ça casserait le chargement des
> checkpoints v2). Défauts actuels : Débutant=v3, Intermédiaire=v3,
> Avancé=v2, Expert=v2+v3 (toutes les deux exposées sur le curseur).

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

Benchmarks (150 parties vs aléatoire, seed=42) : Débutant 68,7% win /
43,53 score moyen ; Intermédiaire 90,7% / 32,90 ; Avancé 97,3% / 24,30 ;
Expert 98,7% / 19,89. Face-à-face 1v1 vs Expert v2 (200 parties) : v3
gagne 57,0%, v2 41,0%, 2 égalités.

## Lignée v4 (`OBS_DIM = 1701`, tête auxiliaire de classement)

Run `train_v4` : warm-start depuis les poids v3 Expert
(`checkpoints/skynet_v3.pt` copié vers `checkpoints/skynet_v4.pt` puis
`--resume`, tronc/politique/valeur repris tels quels, seule la nouvelle
tête `rank_head` — ajoutée dans `network.py` — part de zéro), 700
itérations, entropie basse (0.004→0.0008), perte auxiliaire de
classement (`--rank-coef 0.1`, cross-entropie sur le rang final,
masquée selon le nombre de joueurs). Un seul niveau produit (Expert),
pas de ladder Débutant→Avancé pour cette lignée.

| Fichier (`checkpoints/levels_v4/`) | Correspondrait à | Origine |
|---|---|---|
| `level_4.pt` | Expert | Run `train_v4`, itération 700/700 (warm-start depuis v3 Expert) |

Benchmarks : 99,3% win vs aléatoire / score moyen 18,49. Face-à-face 1v1
vs Expert v3 (200 parties) : v4 gagne 56,5%, v3 43,0%, 1 égalité — gain
du même ordre que v3 vs v2, la tête de classement apporte un progrès
mesurable mais modeste.

Pour ajouter un niveau : relancer `train.py` avec `--milestones` pointant
sur les itérations voulues, ou copier `checkpoints/skynet.pt` vers le
fichier de niveau cible une fois l'entraînement terminé — et mettre à
jour ce tableau et `VARIANTS` dans `server.py`.
