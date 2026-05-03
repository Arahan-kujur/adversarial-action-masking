"""Budget sweep: learned vs value heuristic with 10 seeds and CIs.

Addresses reviewer concern: at what budget does learned > heuristic?
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
SEEDS = list(range(10))


def value_heuristic_mask(agent, budget):
    q_importance = {}
    for s in P0_STATES:
        if s in agent.q:
            q_importance[s] = float(np.max(np.abs(agent.q[s])))
        else:
            q_importance[s] = 0.0
    ranked = sorted(P0_STATES, key=lambda s: q_importance[s], reverse=True)
    targets = set(ranked[:budget])

    def mask_fn(info_state, legal_actions, player):
        if player != 0 or len(legal_actions) <= 1:
            return legal_actions
        if info_state not in targets:
            return legal_actions
        best = max(legal_actions, key=lambda a: agent.q[info_state][a])
        filtered = [a for a in legal_actions if a != best]
        return filtered if filtered else legal_actions
    return mask_fn


def run_one(budget, mode, seed):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

    for _ in range(10000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    if budget == 0:
        return evaluate_agent(env, agent, rng, 5000)

    if mode == "learned":
        adv = AdversarialMask(target_player=0, num_actions=2, lr=0.01,
                              budget=budget)
        env.set_mask(adv.mask_fn)
        for o in range(30):  # more outer iterations for convergence
            batch = []
            for _ in range(500):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        return evaluate_agent(env, agent, rng, 5000, mask_fn=adv.mask_fn)

    elif mode == "value":
        m = value_heuristic_mask(agent, budget)
        env.set_mask(m)
        for _ in range(15000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        return evaluate_agent(env, agent, rng, 5000, mask_fn=m)

    elif mode == "random":
        prob = budget / 6.0
        rm = random_mask(0, prob, np.random.default_rng(seed + 99))
        env.set_mask(rm)
        for _ in range(15000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        return evaluate_agent(env, agent, rng, 5000, mask_fn=rm)


def main():
    budgets = [1, 2, 3, 4, 5, 6]

    print("Budget Sweep: Learned vs Value Heuristic vs Random (10 seeds)")
    print(f"{'Budget':>6s}  {'Learned':>20s}  {'Value Heur.':>20s}  {'Random':>20s}")
    print("-" * 72)

    for budget in budgets:
        results = {}
        for mode in ["learned", "value", "random"]:
            vals = [run_one(budget, mode, s) for s in SEEDS]
            m = np.mean(vals)
            ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
            results[mode] = (m, ci)

        print(f"{budget:>6d}  "
              f"{results['learned'][0]:+.4f}+/-{results['learned'][1]:.4f}  "
              f"{results['value'][0]:+.4f}+/-{results['value'][1]:.4f}  "
              f"{results['random'][0]:+.4f}+/-{results['random'][1]:.4f}")


if __name__ == "__main__":
    main()
