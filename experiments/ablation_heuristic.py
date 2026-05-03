"""Ablation: learned adversary vs heuristic adversaries.

Compares:
  1. Learned adversary (bi-level trained)
  2. Frequency heuristic (remove at most-visited states)
  3. Value heuristic (remove at highest-value states)
  4. Random baseline

All with same budget (mask at 3/6 states).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask
from adversary.mask_utils import evaluate_agent


P0_STATES = ["0", "0pb", "1", "1pb", "2", "2pb"]
BUDGET = 3


def compute_heuristic_mask(agent, env, rng, heuristic="frequency",
                           budget=BUDGET, num_episodes=5000):
    """Build a mask targeting top-budget states by a heuristic."""
    visit_counts = {s: 0 for s in P0_STATES}
    value_sums = {s: [] for s in P0_STATES}

    for _ in range(num_episodes):
        env.reset(rng=rng)
        traj = []
        while not env.is_terminal:
            player = env.current_player
            info = env.info_state(player)
            legal = env.legal_actions()
            action = agent.select_action(info, legal, rng)
            traj.append((player, info, action))
            env.step(action)
        reward = env.returns[0]
        for player, info, action in traj:
            if player == 0 and info in visit_counts:
                visit_counts[info] += 1
                value_sums[info].append(reward)

    if heuristic == "frequency":
        ranked = sorted(P0_STATES, key=lambda s: visit_counts[s], reverse=True)
    elif heuristic == "value":
        avg_val = {s: np.mean(v) if v else 0 for s, v in value_sums.items()}
        ranked = sorted(P0_STATES, key=lambda s: abs(avg_val[s]), reverse=True)
    else:
        ranked = P0_STATES

    target_states = set(ranked[:budget])

    def mask_fn(info_state, legal_actions, player):
        if player != 0 or len(legal_actions) <= 1:
            return legal_actions
        if info_state not in target_states:
            return legal_actions
        q_vals = agent.q[info_state]
        best_action = max(legal_actions, key=lambda a: q_vals[a])
        filtered = [a for a in legal_actions if a != best_action]
        return filtered if filtered else legal_actions

    return mask_fn, target_states


def run_ablation(seeds=[42, 123, 456, 789, 1024], pretrain=10000,
                 attack_eps=10000, eval_ep=5000):
    print("=" * 60)
    print("  Ablation: Learned vs Heuristic Adversaries")
    print(f"  Budget: {BUDGET}/{len(P0_STATES)} states")
    print("=" * 60)

    results = {}

    for label in ["Learned", "Frequency heuristic", "Value heuristic", "Random"]:
        vals = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            env = MaskedKuhnPoker(KuhnPokerEnv())
            agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

            for _ in range(pretrain):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)

            if label == "Learned":
                adversary = AdversarialMask(target_player=0, num_actions=2,
                                            lr=0.01, budget=BUDGET)
                env.set_mask(adversary.mask_fn)
                for outer in range(20):
                    batch = []
                    for _ in range(attack_eps // 20):
                        r, t = play_episode(env, agent, rng)
                        agent.update(t, r)
                        batch.append((t, r))
                    adversary.update(batch)
                r = evaluate_agent(env, agent, rng, eval_ep,
                                   mask_fn=adversary.mask_fn)

            elif label == "Random":
                rmask = random_mask(0, BUDGET / len(P0_STATES),
                                    np.random.default_rng(seed + 999))
                env.set_mask(rmask)
                for _ in range(attack_eps):
                    r_, t = play_episode(env, agent, rng)
                    agent.update(t, r_)
                r = evaluate_agent(env, agent, rng, eval_ep, mask_fn=rmask)

            else:
                heuristic = "frequency" if "Frequency" in label else "value"
                hmask, targets = compute_heuristic_mask(
                    agent, env, np.random.default_rng(seed), heuristic, BUDGET)
                env.set_mask(hmask)
                for _ in range(attack_eps):
                    r_, t = play_episode(env, agent, rng)
                    agent.update(t, r_)
                r = evaluate_agent(env, agent, rng, eval_ep, mask_fn=hmask)

            vals.append(r)

        mean = np.mean(vals)
        std = np.std(vals)
        results[label] = (mean, std)
        print(f"  {label:<25s}: {mean:+.4f} +/- {std:.4f}")

    print("\n  Lower = more effective attack.")


if __name__ == "__main__":
    run_ablation()
