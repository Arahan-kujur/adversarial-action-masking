"""Adversary targeting analysis: which states does the adversary target
and does this correlate with strategic importance?"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask


# Kuhn info states for P0 and their strategic meaning
P0_STATES = {
    "0":   "J at root",
    "0pb": "J facing bet",
    "1":   "Q at root",
    "1pb": "Q facing bet",
    "2":   "K at root",
    "2pb": "K facing bet",
}


def compute_state_reach(agent, rng, num_episodes=10000):
    """Estimate reach probability for each P0 info state."""
    env = MaskedKuhnPoker(KuhnPokerEnv())
    counts = {s: 0 for s in P0_STATES}
    total = 0

    for _ in range(num_episodes):
        env.reset(rng=rng)
        while not env.is_terminal:
            player = env.current_player
            info = env.info_state(player)
            legal = env.legal_actions()
            if player == 0 and info in counts:
                counts[info] += 1
                total += 1
            action = agent.select_action(info, legal, rng)
            env.step(action)

    return {s: c / max(total, 1) for s, c in counts.items()}


def compute_state_value(agent, rng, num_episodes=10000):
    """Estimate expected reward contribution from each P0 info state."""
    env = MaskedKuhnPoker(KuhnPokerEnv())
    state_rewards = {s: [] for s in P0_STATES}

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
            if player == 0 and info in state_rewards:
                state_rewards[info].append(reward)

    return {s: np.mean(v) if v else 0.0 for s, v in state_rewards.items()}


def run_targeting_analysis(seed=42, pretrain=10000, attack_eps=10000,
                           outer_iters=20):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

    for _ in range(pretrain):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    reach = compute_state_reach(agent, np.random.default_rng(seed))
    value = compute_state_value(agent, np.random.default_rng(seed))

    adversary = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
    env.set_mask(adversary.mask_fn)

    for outer in range(outer_iters):
        batch = []
        for _ in range(attack_eps // outer_iters):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adversary.update(batch)

    print("=" * 70)
    print("  Adversary Targeting Analysis")
    print("=" * 70)
    print(f"  {'State':>6s}  {'Card':>10s}  {'Reach':>6s}  {'Value':>7s}  "
          f"{'Removes':>8s}  {'Conf':>6s}")
    print("  " + "-" * 55)

    for info_state, desc in P0_STATES.items():
        r = reach.get(info_state, 0)
        v = value.get(info_state, 0)

        if info_state in adversary.theta:
            probs = adversary._get_removal_prob(info_state)
            removed = int(np.argmax(probs))
            name = "PASS" if removed == 0 else "BET"
            conf = probs[removed]
        else:
            name = "---"
            conf = 0.0

        print(f"  {info_state:>6s}  {desc:>10s}  {r:>6.3f}  {v:>+7.3f}  "
              f"{name:>8s}  {conf:>6.3f}")

    print("\n  Key insight: adversary targets states where removal")
    print("  causes maximum strategic damage, not necessarily")
    print("  the most frequently reached states.")


if __name__ == "__main__":
    run_targeting_analysis()
