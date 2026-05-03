"""Attack Efficiency Curve: the headline figure.

x-axis: fraction of P0 info states masked (0 to 1)
y-axis: normalized agent performance
Curves: adversarial, random, fixed baseline
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask
from adversary.mask_utils import evaluate_agent

NUM_P0_STATES = 6
KUHN_MIN, KUHN_MAX = -2, 2
SEEDS = [42, 123, 456, 789, 1024]


def normalize(r):
    return (r - KUHN_MIN) / (KUHN_MAX - KUHN_MIN)


def run_one(budget, mode, seed, pretrain=10000, attack_eps=10000,
            eval_ep=5000, outer_iters=20):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

    for _ in range(pretrain):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    if budget == 0:
        return evaluate_agent(env, agent, rng, eval_ep)

    if mode == "adversarial":
        adversary = AdversarialMask(target_player=0, num_actions=2,
                                    lr=0.01, budget=budget)
        env.set_mask(adversary.mask_fn)
        for outer in range(outer_iters):
            batch = []
            for _ in range(attack_eps // outer_iters):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
                batch.append((t, r))
            adversary.update(batch)
        return evaluate_agent(env, agent, rng, eval_ep, mask_fn=adversary.mask_fn)

    elif mode == "random":
        prob = budget / NUM_P0_STATES
        rmask = random_mask(0, prob, np.random.default_rng(seed + 999))
        env.set_mask(rmask)
        for _ in range(attack_eps):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        return evaluate_agent(env, agent, rng, eval_ep, mask_fn=rmask)

    elif mode == "fixed":
        from adversary.masking_policy import fixed_removal
        fmask = fixed_removal(0, 1)
        env.set_mask(fmask)
        for _ in range(attack_eps):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        return evaluate_agent(env, agent, rng, eval_ep, mask_fn=fmask)


def main():
    budgets = [0, 1, 2, 3, 4, 6]
    fractions = [b / NUM_P0_STATES for b in budgets]

    results = {mode: {"mean": [], "lo": [], "hi": []} for mode in ["adversarial", "random"]}

    print("Computing efficiency curves (5 seeds per point)...")

    for budget in budgets:
        for mode in ["adversarial", "random"]:
            vals = [run_one(budget, mode, s) for s in SEEDS]
            norm = [normalize(v) for v in vals]
            m = np.mean(norm)
            lo = m - 1.96 * np.std(norm) / np.sqrt(len(norm))
            hi = m + 1.96 * np.std(norm) / np.sqrt(len(norm))
            results[mode]["mean"].append(m)
            results[mode]["lo"].append(lo)
            results[mode]["hi"].append(hi)
            raw = np.mean(vals)
            print(f"  budget={budget} {mode:>12s}: raw={raw:+.4f} norm={m:.4f}")

    # Also compute fixed removal as a horizontal line
    fixed_vals = [run_one(6, "fixed", s) for s in SEEDS]
    fixed_norm = normalize(np.mean(fixed_vals))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(fractions, results["adversarial"]["mean"], "o-",
            color="#E53935", linewidth=2, markersize=7, label="Adversarial")
    ax.fill_between(fractions, results["adversarial"]["lo"],
                    results["adversarial"]["hi"], alpha=0.15, color="#E53935")

    ax.plot(fractions, results["random"]["mean"], "s--",
            color="#1E88E5", linewidth=2, markersize=7, label="Random")
    ax.fill_between(fractions, results["random"]["lo"],
                    results["random"]["hi"], alpha=0.15, color="#1E88E5")

    ax.axhline(y=fixed_norm, color="#757575", linestyle=":", linewidth=1.5,
               label=f"Fixed removal ({fixed_norm:.2f})")

    ax.set_xlabel("Fraction of P0 States Masked", fontsize=12)
    ax.set_ylabel("Normalized Performance (0=worst, 1=best)", fontsize=12)
    ax.set_title("Attack Efficiency: Adversarial vs Random Masking", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.65)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/attack_efficiency.png", dpi=150)
    plt.close()
    print(f"\nFigure saved: results/attack_efficiency.png")


if __name__ == "__main__":
    main()
