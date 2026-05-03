"""NFSP victim experiment: does NFSP resist adversarial masking?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.nfsp import NFSPAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask
from adversary.mask_utils import evaluate_agent

SEEDS = list(range(10))


def run_one(mode, seed):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = NFSPAgent(num_actions=2, alpha=0.1, epsilon=0.15, eta=0.1)

    for _ in range(10000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    pre_reward = evaluate_agent(env, agent, rng, 5000)

    if mode == "none":
        return pre_reward

    adv = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
    env.set_mask(adv.mask_fn)
    for o in range(20):
        batch = []
        for _ in range(500):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)

    post_reward = evaluate_agent(env, agent, rng, 5000, mask_fn=adv.mask_fn)
    return post_reward


def main():
    print("NFSP Victim Under Adversarial Masking (10 seeds)")
    print(f"{'Mode':>15s}  {'Mean':>8s}  {'CI':>8s}")
    print("-" * 40)
    for mode in ["none", "adversarial"]:
        vals = [run_one(mode, s) for s in SEEDS]
        m = np.mean(vals)
        ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
        print(f"{mode:>15s}  {m:+.4f}  +/-{ci:.4f}")


if __name__ == "__main__":
    main()
