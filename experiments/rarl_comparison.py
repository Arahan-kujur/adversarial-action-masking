"""RARL comparison: action perturbation vs action removal."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask
from adversary.mask_utils import evaluate_agent

SEEDS = list(range(10))
PRETRAIN = 10000
ATTACK_EPS = 15000
EVAL_EPS = 5000


def rarl_episode(env, agent, rng, perturb_prob=0.3):
    """Play episode with RARL-style action perturbation (worst-action override)."""
    env.reset(rng=rng)
    trajectory = []
    while not env.is_terminal:
        player = env.current_player
        info = env.info_state(player)
        legal = env.legal_actions()
        action = agent.select_action(info, legal, rng)
        if player == 0 and rng.random() < perturb_prob and len(legal) > 1:
            q_vals = agent.q[info]
            action = min(legal, key=lambda a: q_vals[a])
        env.step(action)
        trajectory.append((player, info, action))
    return env.returns[0], trajectory


def run_one(mode, seed):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

    for _ in range(PRETRAIN):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    if mode == "none":
        return evaluate_agent(env, agent, rng, EVAL_EPS)

    elif mode == "rarl":
        for _ in range(ATTACK_EPS):
            r, t = rarl_episode(env, agent, rng, perturb_prob=0.3)
            agent.update(t, r)
        rewards = []
        for _ in range(EVAL_EPS):
            r, t = rarl_episode(env, agent, rng, perturb_prob=0.3)
            rewards.append(r)
        return np.mean(rewards)

    elif mode == "removal":
        adv = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
        env.set_mask(adv.mask_fn)
        for o in range(20):
            batch = []
            for _ in range(500):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        return evaluate_agent(env, agent, rng, EVAL_EPS, mask_fn=adv.mask_fn)


def main():
    print("RARL (action perturbation) vs Action Removal (10 seeds)")
    print(f"{'Mode':>25s}  {'Mean':>8s}  {'CI':>8s}")
    print("-" * 50)
    for mode in ["none", "rarl", "removal"]:
        vals = [run_one(mode, s) for s in SEEDS]
        m = np.mean(vals)
        ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
        print(f"{mode:>25s}  {m:+.4f}  +/-{ci:.4f}")


if __name__ == "__main__":
    main()
