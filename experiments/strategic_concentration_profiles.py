"""Strategic concentration profiles for Leduc-N self-play policies.

Computes attacker-facing dependence profiles P_pi(k), C_k, Gini, entropy,
SRI, and action-removal damage across Leduc-N scales, checkpoints, budgets,
and victim classes. This script is intentionally standalone so it can reuse the
Paper 2 environments and agents without modifying their public interfaces.
"""

import argparse
import copy
import csv
import math
import os
import sys
from pathlib import Path
from types import MethodType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch

from adversary.mask_utils import evaluate_agent
from adversary.masking_policy import random_mask
from core.agents.dqn import DQNAgent
from core.agents.q_learning import QLearningAgent
from core.envs.leduc_n import LeducNPokerEnv, MaskedLeducNPoker, NUM_ACTIONS
from core.training.selfplay import play_episode


class EntropyQLearningAgent(QLearningAgent):
    def __init__(self, *args, temperature=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def select_action(self, info_state, legal_actions, rng):
        q_vals = self.q[info_state]
        logits = np.array([q_vals[a] for a in legal_actions], dtype=float) / max(self.temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return int(rng.choice(legal_actions, p=probs))

    def policy_probs(self, info_state, legal_actions):
        probs = np.zeros(self.num_actions)
        q_vals = self.q[info_state]
        logits = np.array([q_vals[a] for a in legal_actions], dtype=float) / max(self.temperature, 1e-6)
        logits -= logits.max()
        local = np.exp(logits)
        local /= local.sum()
        for action, prob in zip(legal_actions, local):
            probs[action] = prob
        return probs


class CRTQLearningAgent(QLearningAgent):
    def __init__(self, *args, crt_lambda=0.08, sri_epsilon=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.crt_lambda = crt_lambda
        self.sri_epsilon = sri_epsilon

    def update(self, trajectory, reward_p0):
        super().update(trajectory, reward_p0)
        touched = {(info_state, tuple(sorted({action for p, s, action in trajectory if s == info_state})))
                   for player, info_state, _ in trajectory if player == 0}
        for info_state, seen_actions in touched:
            legal = list(seen_actions)
            if len(legal) < 2:
                legal = list(range(self.num_actions))
            q_vals = self.q[info_state]
            best = max(legal, key=lambda a: q_vals[a])
            floor = q_vals[best] - self.sri_epsilon
            for action in legal:
                if action != best and q_vals[action] < floor:
                    q_vals[action] += self.crt_lambda * (floor - q_vals[action])


def make_dqn(game_name, epsilon_start=1.0):
    return DQNAgent(
        game=game_name,
        num_actions=NUM_ACTIONS,
        lr=1e-3,
        epsilon_start=epsilon_start,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        buffer_size=20000,
        batch_size=64,
        target_update_freq=500,
    )


def make_entropy_dqn(game_name, temperature=0.7):
    agent = make_dqn(game_name, epsilon_start=0.05)

    def select_action(self, info_state, legal_actions, rng):
        with torch.no_grad():
            q_vals = self._online(self._encode(info_state)).squeeze(0).cpu().numpy()
        logits = np.array([q_vals[a] for a in legal_actions], dtype=float) / max(temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return int(rng.choice(legal_actions, p=probs))

    def policy_probs(self, info_state, legal_actions):
        probs = np.zeros(self.num_actions)
        with torch.no_grad():
            q_vals = self._online(self._encode(info_state)).squeeze(0).cpu().numpy()
        logits = np.array([q_vals[a] for a in legal_actions], dtype=float) / max(temperature, 1e-6)
        logits -= logits.max()
        local = np.exp(logits)
        local /= local.sum()
        for action, prob in zip(legal_actions, local):
            probs[action] = prob
        return probs

    agent.select_action = MethodType(select_action, agent)
    agent.policy_probs = MethodType(policy_probs, agent)
    return agent


def make_agent(victim, method, ranks, args):
    game_name = f"leduc{ranks}"
    if victim == "q":
        if method == "entropy":
            return EntropyQLearningAgent(num_actions=NUM_ACTIONS, alpha=args.alpha, epsilon=args.epsilon,
                                         temperature=args.entropy_temp)
        if method == "crt_sri":
            return CRTQLearningAgent(num_actions=NUM_ACTIONS, alpha=args.alpha, epsilon=args.epsilon,
                                     crt_lambda=args.crt_lambda, sri_epsilon=args.sri_epsilon)
        return QLearningAgent(num_actions=NUM_ACTIONS, alpha=args.alpha, epsilon=args.epsilon)
    if victim == "dqn":
        if method == "entropy":
            return make_entropy_dqn(game_name, temperature=args.entropy_temp)
        return make_dqn(game_name)
    raise ValueError(f"unknown victim {victim}")


def q_values(agent, info_state):
    if hasattr(agent, "q"):
        return np.asarray(agent.q[info_state], dtype=float)
    with torch.no_grad():
        return agent._online(agent._encode(info_state)).squeeze(0).cpu().numpy().astype(float)


def train_to_checkpoint(env, agent, rng, current, target):
    for _ in range(current, target):
        reward, trajectory = play_episode(env, agent, rng)
        agent.update(trajectory, reward)
    return target


def collect_reach_and_legals(env, agent, rng, episodes):
    counts = {}
    legal_by_state = {}
    for _ in range(episodes):
        env.reset(rng=rng)
        while not env.is_terminal:
            player = env.current_player
            info = env.info_state(player)
            legal = env.legal_actions()
            if player == 0:
                counts[info] = counts.get(info, 0) + 1
                legal_by_state.setdefault(info, tuple(legal))
            action = agent.select_action(info, legal, rng)
            env.step(action)
    total = max(sum(counts.values()), 1)
    reach = {state: count / total for state, count in counts.items()}
    return reach, legal_by_state


def dependence_scores(agent, reach, legal_by_state):
    rows = []
    for state, rho in reach.items():
        legal = list(legal_by_state[state])
        if len(legal) < 2:
            continue
        values = q_values(agent, state)
        legal_values = sorted((float(values[a]) for a in legal), reverse=True)
        gap = legal_values[0] - legal_values[1]
        rows.append((state, rho, gap, rho * gap, legal))
    rows.sort(key=lambda row: row[3], reverse=True)
    return rows


def gini(values):
    if not values:
        return 0.0
    vals = sorted(max(0.0, float(v)) for v in values)
    total = sum(vals)
    if total <= 0:
        return 0.0
    n = len(vals)
    weighted = sum((i + 1) * value for i, value in enumerate(vals))
    return (2 * weighted) / (n * total) - (n + 1) / n


def entropy_metric(agent, reach, legal_by_state):
    total = 0.0
    for state, rho in reach.items():
        legal = list(legal_by_state[state])
        probs = agent.policy_probs(state, legal)
        total += rho * (-sum(float(probs[a]) * math.log(max(float(probs[a]), 1e-12)) for a in legal))
    return total


def sri_metric(agent, reach, legal_by_state, eps):
    total = 0.0
    for state, rho in reach.items():
        legal = list(legal_by_state[state])
        values = q_values(agent, state)
        best = max(float(values[a]) for a in legal)
        near = [a for a in legal if best - float(values[a]) <= eps]
        total += rho * math.log(max(len(near), 1))
    return total


def profile_metrics(score_rows, budgets):
    scores = [max(0.0, row[3]) for row in score_rows]
    total = sum(scores)
    out = {
        "dependence_total": total,
        "dependence_gini": gini(scores),
    }
    for k in [1, 2, 4, 8, 16, 32]:
        ck = sum(scores[:k])
        out[f"c{k}"] = ck
        out[f"p{k}"] = ck / total if total > 0 else 0.0
    for budget in budgets:
        ck = sum(scores[:budget])
        out[f"ck_budget_{budget}"] = ck
        out[f"p_budget_{budget}"] = ck / total if total > 0 else 0.0
    return out


def gap_attack_mask(agent, score_rows, budget):
    active = {state for state, *_ in score_rows[:budget]}
    remove = {}
    for state, _, _, _, legal in score_rows[:budget]:
        values = q_values(agent, state)
        remove[state] = max(legal, key=lambda a: float(values[a]))

    def mask_fn(info_state, legal_actions, player):
        if player != 0 or info_state not in active:
            return legal_actions
        filtered = [a for a in legal_actions if a != remove[info_state]]
        return filtered if filtered else legal_actions

    return mask_fn


def train_learned_adversary(env, agent, rng, budget, iterations, inner_episodes):
    from adversary.masking_policy import AdversarialMask
    adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01, budget=budget)
    env.set_mask(adv.mask_fn)
    for _ in range(iterations):
        batch = []
        for _ in range(inner_episodes):
            reward, trajectory = play_episode(env, agent, rng)
            agent.update(trajectory, reward)
            batch.append((trajectory, reward))
        adv.update(batch)
    env.set_mask(None)
    return adv.mask_fn


def measure(env, agent, rng, ranks, victim, method, checkpoint, budget, seed, args):
    reach, legal_by_state = collect_reach_and_legals(env, agent, rng, args.reach_episodes)
    scores = dependence_scores(agent, reach, legal_by_state)
    profile = profile_metrics(scores, args.budgets)
    clean = evaluate_agent(env, agent, rng, args.eval_episodes)
    random_fn = random_mask(0, args.random_remove_prob, np.random.default_rng(seed + 12345 + budget))
    random_reward = evaluate_agent(env, agent, rng, args.eval_episodes, mask_fn=random_fn)
    gap_fn = gap_attack_mask(agent, scores, min(budget, len(scores)))
    gap_reward = evaluate_agent(env, agent, rng, args.eval_episodes, mask_fn=gap_fn)
    learned_reward = float("nan")
    if args.learned_adversary:
        learned_env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=ranks))
        learned_agent = copy.deepcopy(agent)
        learned_fn = train_learned_adversary(learned_env, learned_agent, rng, budget,
                                             args.learned_iters, args.learned_inner)
        learned_reward = evaluate_agent(env, agent, rng, args.eval_episodes, mask_fn=learned_fn)
    row = {
        "ranks": ranks,
        "environment": f"leduc{ranks}",
        "victim": victim,
        "method": method,
        "checkpoint": checkpoint,
        "budget": budget,
        "seed": seed,
        "clean_reward": clean,
        "random_reward": random_reward,
        "gap_reward": gap_reward,
        "learned_reward": learned_reward,
        "random_damage": clean - random_reward,
        "gap_damage": clean - gap_reward,
        "learned_damage": clean - learned_reward if not math.isnan(learned_reward) else float("nan"),
        "entropy": entropy_metric(agent, reach, legal_by_state),
        "sri": sri_metric(agent, reach, legal_by_state, args.sri_epsilon),
        "visited_states": len(reach),
    }
    row.update(profile)
    return row, scores


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if not math.isnan(float(x)) and not math.isnan(float(y))]
    if len(pairs) < 2:
        return float("nan")
    x_vals, y_vals = zip(*pairs)
    return float(np.corrcoef(x_vals, y_vals)[0, 1])


def summarize_correlations(rows):
    metrics = ["p1", "p2", "p4", "p8", "p16", "dependence_gini", "dependence_total", "entropy", "sri", "clean_reward"]
    targets = ["gap_damage", "random_damage"]
    if any(not math.isnan(float(row["learned_damage"])) for row in rows):
        targets.append("learned_damage")
    scopes = [("all", rows)]
    for victim in sorted({row["victim"] for row in rows}):
        scopes.append((f"victim={victim}", [row for row in rows if row["victim"] == victim]))
    for ranks in sorted({int(row["ranks"]) for row in rows}):
        scopes.append((f"leduc{ranks}", [row for row in rows if int(row["ranks"]) == ranks]))
    out = []
    for scope, subset in scopes:
        for metric in metrics:
            for target in targets:
                out.append({
                    "scope": scope,
                    "metric": metric,
                    "target": target,
                    "pearson": pearson([row[metric] for row in subset], [row[target] for row in subset]),
                    "n": len(subset),
                })
    return out


def write_lorenz(path, snapshots):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "rank_fraction", "mass_fraction"])
        writer.writeheader()
        for label, scores in snapshots:
            vals = [max(0.0, row[3]) for row in scores]
            total = sum(vals)
            if total <= 0:
                continue
            cumulative = 0.0
            for i, value in enumerate(vals, start=1):
                cumulative += value
                writer.writerow({"label": label, "rank_fraction": i / len(vals), "mass_fraction": cumulative / total})


def plot_outputs(outdir, rows, corr_rows):
    outdir.mkdir(parents=True, exist_ok=True)
    damage = np.array([float(row["gap_damage"]) for row in rows])
    for metric in ["p1", "dependence_gini", "entropy", "sri"]:
        xs = np.array([float(row[metric]) for row in rows])
        plt.figure(figsize=(4.5, 3.5))
        plt.scatter(xs, damage, alpha=0.7, s=18)
        plt.xlabel(metric)
        plt.ylabel("Gap-attack damage")
        plt.title(f"{metric} vs damage")
        plt.tight_layout()
        plt.savefig(outdir / f"{metric}_vs_damage.png", dpi=200)
        plt.close()

    top = [row for row in corr_rows if row["scope"] == "all" and row["target"] == "gap_damage"]
    top = sorted(top, key=lambda row: abs(float(row["pearson"])), reverse=True)
    with (outdir / "top_correlations.txt").open("w", encoding="utf-8") as f:
        for row in top:
            f.write(f"{row['metric']}: r={float(row['pearson']):.3f} n={row['n']}\n")


def run(args):
    rows = []
    snapshots = []
    for ranks in args.ranks:
        for victim in args.victims:
            methods = list(args.methods)
            if victim == "dqn":
                methods = [m for m in methods if m != "crt_sri"]
            for method in methods:
                for seed_idx in range(args.seeds):
                    seed = args.seed + ranks * 10000 + seed_idx * 1000 + sum(ord(c) for c in victim + method)
                    rng = np.random.default_rng(seed)
                    torch.manual_seed(seed)
                    env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=ranks))
                    agent = make_agent(victim, method, ranks, args)
                    current = 0
                    for checkpoint in sorted(args.checkpoints):
                        print(f"run ranks={ranks} victim={victim} method={method} seed={seed_idx} checkpoint={checkpoint}", flush=True)
                        current = train_to_checkpoint(env, agent, rng, current, checkpoint)
                        for budget in args.budgets:
                            row, scores = measure(env, agent, rng, ranks, victim, method, checkpoint, budget, seed_idx, args)
                            rows.append(row)
                            if budget == args.budgets[0] and len(snapshots) < args.max_lorenz_snapshots:
                                label = f"L{ranks}-{victim}-{method}-s{seed_idx}-t{checkpoint}"
                                snapshots.append((label, scores))
    return rows, snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", nargs="+", type=int, default=[3, 5, 10, 20])
    parser.add_argument("--victims", nargs="+", default=["dqn"])
    parser.add_argument("--methods", nargs="+", default=["standard", "entropy", "crt_sri"])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[1000, 4000, 10000])
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--entropy-temp", type=float, default=0.7)
    parser.add_argument("--crt-lambda", type=float, default=0.08)
    parser.add_argument("--sri-epsilon", type=float, default=0.25)
    parser.add_argument("--reach-episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--random-remove-prob", type=float, default=0.5)
    parser.add_argument("--learned-adversary", action="store_true")
    parser.add_argument("--learned-iters", type=int, default=5)
    parser.add_argument("--learned-inner", type=int, default=100)
    parser.add_argument("--max-lorenz-snapshots", type=int, default=12)
    parser.add_argument("--outdir", type=Path, default=Path("results/strategic_concentration_profiles"))
    args = parser.parse_args()

    rows, snapshots = run(args)
    corr_rows = summarize_correlations(rows)
    write_csv(args.outdir / "raw.csv", rows)
    write_csv(args.outdir / "correlations.csv", corr_rows)
    write_lorenz(args.outdir / "lorenz_curves.csv", snapshots)
    plot_outputs(args.outdir / "figures", rows, corr_rows)
    print(f"Wrote strategic concentration outputs to {args.outdir}")


if __name__ == "__main__":
    main()

