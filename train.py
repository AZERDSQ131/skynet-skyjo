"""Entraînement self-play PPO pour Skynet (Skyjo)."""

import argparse
import os
import random
import time

import numpy as np
import torch

from skynet.agents.network import ActorCriticNet
from skynet.agents.ppo import ppo_update
from skynet.agents.rollout import collect_rollout
from skynet.env.skyjo_env import OBS_DIM, N_ACTIONS


def evaluate_vs_random(policy, device, n_episodes=50, rng=None):
    """Le joueur 0 utilise la politique, les autres jouent aléatoirement."""
    from skynet.env.skyjo_env import SkyjoEnv

    rng = rng or random.Random()
    wins, placements, own_scores = 0, [], []
    for _ in range(n_episodes):
        env = SkyjoEnv(rng=rng)
        while not env.done:
            player = env.current_player_id()
            mask = env.legal_action_mask()
            if player == 0:
                obs_vec = env.observe(player)
                obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                action, _, _ = policy.act(obs_t, mask_t)
                a = int(action.item())
            else:
                legal = np.nonzero(mask)[0]
                a = int(rng.choice(legal))
            env.step(a)
        scores = env.scores
        own_scores.append(scores[0])
        rank = sorted(scores).index(scores[0])
        placements.append(rank)
        if scores[0] == min(scores):
            wins += 1
    return {
        "win_rate": wins / n_episodes,
        "avg_score": float(np.mean(own_scores)),
        "avg_placement": float(np.mean(placements)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--episodes-per-iter", type=int, default=48)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--entropy-start", type=float, default=0.02)
    parser.add_argument("--entropy-end", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/skynet.pt")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--levels-dir", type=str, default="checkpoints/levels")
    parser.add_argument("--milestones", type=str, default="150,500,1200,3000")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device)
    policy = ActorCriticNet(OBS_DIM, N_ACTIONS).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    os.makedirs(args.levels_dir, exist_ok=True)

    if args.resume and os.path.exists(args.checkpoint):
        try:
            policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Reprise depuis {args.checkpoint}")
        except RuntimeError as e:
            print(f"Impossible de reprendre ({e}); démarrage à neuf avec la nouvelle architecture.")

    milestones = sorted(int(m) for m in args.milestones.split(",") if m.strip())
    milestone_idx = 0

    for it in range(1, args.iterations + 1):
        t0 = time.time()

        progress = min(1.0, it / args.iterations)
        ent_coef = args.entropy_start + (args.entropy_end - args.entropy_start) * progress

        samples, scores, n_players = collect_rollout(
            policy, device, args.episodes_per_iter, args.gamma, args.lam, rng
        )
        stats = ppo_update(
            policy, optimizer, samples, device,
            clip_eps=args.clip_eps, epochs=args.epochs, batch_size=args.batch_size,
            ent_coef=ent_coef,
        )
        dt = time.time() - t0

        avg_score = float(np.mean([s for ep_scores in scores for s in ep_scores]))
        print(
            f"it={it:5d} | ep/iter={args.episodes_per_iter} | "
            f"avg_score={avg_score:6.2f} | policy_loss={stats['policy_loss']:+.4f} | "
            f"value_loss={stats['value_loss']:.4f} | entropy={stats['entropy']:.3f} | "
            f"ent_coef={ent_coef:.4f} | {dt:.1f}s"
        )

        if it % args.eval_every == 0:
            eval_stats = evaluate_vs_random(policy, device, args.eval_episodes, rng)
            print(
                f"  [eval vs random] win_rate={eval_stats['win_rate']:.2%} "
                f"avg_score={eval_stats['avg_score']:.2f} "
                f"avg_placement={eval_stats['avg_placement']:.2f}"
            )

        if it % args.checkpoint_every == 0:
            torch.save(policy.state_dict(), args.checkpoint)
            print(f"  checkpoint sauvegardé -> {args.checkpoint}")

        while milestone_idx < len(milestones) and it >= milestones[milestone_idx]:
            level_path = os.path.join(args.levels_dir, f"level_{milestone_idx + 1}.pt")
            torch.save(policy.state_dict(), level_path)
            print(f"  niveau {milestone_idx + 1} sauvegardé -> {level_path} (iteration {it})")
            milestone_idx += 1

    torch.save(policy.state_dict(), args.checkpoint)
    print(f"Entraînement terminé. Checkpoint final -> {args.checkpoint}")


if __name__ == "__main__":
    main()
