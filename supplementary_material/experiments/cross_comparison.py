"""Cross-algorithm + cross-game adversarial attack comparison.

Tests adversarial masking against Q-Learning, PPO, and DQN victims
in both Kuhn and Leduc Poker.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker
from core.agents.q_learning import QLearningAgent
from core.agents.ppo import TabularPPOAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask
from adversary.mask_utils import evaluate_agent


def run_attack(env_cls, masked_cls, num_actions, agent_cls, agent_kwargs,
               seed, pretrain=10000, attack_eps=10000, eval_ep=3000,
               outer_iters=20):
    """Run one adversarial attack experiment. Returns (baseline, attacked)."""
    rng = np.random.default_rng(seed)
    env = masked_cls(env_cls())
    agent = agent_cls(**agent_kwargs)

    for _ in range(pretrain):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    baseline = evaluate_agent(env, agent, rng, eval_ep)

    adversary = AdversarialMask(target_player=0, num_actions=num_actions, lr=0.01)
    env.set_mask(adversary.mask_fn)

    for outer in range(outer_iters):
        batch = []
        for _ in range(attack_eps // outer_iters):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adversary.update(batch)

    attacked = evaluate_agent(env, agent, rng, eval_ep, mask_fn=adversary.mask_fn)
    return baseline, attacked


def main():
    seeds = [42, 123, 456]

    configs = [
        ("Kuhn + Q-Learning", KuhnPokerEnv, MaskedKuhnPoker, 2,
         QLearningAgent, {"num_actions": 2, "alpha": 0.1, "epsilon": 0.15}),
        ("Kuhn + PPO", KuhnPokerEnv, MaskedKuhnPoker, 2,
         TabularPPOAgent, {"num_actions": 2, "lr": 0.01}),
        ("Leduc + Q-Learning", LeducPokerEnv, MaskedLeducPoker, 3,
         QLearningAgent, {"num_actions": 3, "alpha": 0.1, "epsilon": 0.15}),
        ("Leduc + PPO", LeducPokerEnv, MaskedLeducPoker, 3,
         TabularPPOAgent, {"num_actions": 3, "lr": 0.01}),
    ]

    print("=" * 60)
    print("  Cross-Algorithm / Cross-Game Attack Comparison")
    print("=" * 60)
    print(f"  {'Config':<25s}  {'Baseline':>9s}  {'Attacked':>9s}  {'Delta':>8s}")
    print("  " + "-" * 55)

    for name, env_cls, masked_cls, na, agent_cls, kwargs in configs:
        baselines, attackeds = [], []
        for seed in seeds:
            b, a = run_attack(env_cls, masked_cls, na, agent_cls, kwargs, seed)
            baselines.append(b)
            attackeds.append(a)

        bm = np.mean(baselines)
        am = np.mean(attackeds)
        print(f"  {name:<25s}  {bm:>+9.4f}  {am:>+9.4f}  {am-bm:>+8.4f}")


if __name__ == "__main__":
    main()
