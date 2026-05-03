"""Reviewer-response experiments for NeurIPS strengthening.

This script collects compact ablations that address:
1. privileged vs public-information adversary,
2. CACv-greedy oracle baseline,
3. precise L0 budget diagnostics,
4. neural DQN separate-network ablation,
5. stochastic action-dropout defense.

The runs are intentionally moderate so they can be reproduced quickly.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversary.mask_utils import evaluate_agent
from adversary.masking_policy import AdversarialMask, fixed_removal, random_mask
from core.agents.dqn import DQNAgent
from core.agents.q_learning import QLearningAgent
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS
from core.training.selfplay import play_episode


SEEDS = list(range(5))


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(1.96 * values.std() / np.sqrt(len(values)))


def public_key(info_state: str) -> str:
    """Hide private card rank in Leduc info states: rank|public|history -> _|public|history."""
    parts = info_state.split("|")
    if len(parts) >= 3:
        return f"_|{parts[1]}|{parts[2]}"
    return "_|" + "|".join(parts[1:])


class KeyedAdversarialMask(AdversarialMask):
    """AdversarialMask with an information restriction via key_fn."""

    def __init__(self, key_fn, **kwargs):
        super().__init__(**kwargs)
        self.key_fn = key_fn

    def mask_fn(self, info_state, legal_actions, player):
        return super().mask_fn(self.key_fn(info_state), legal_actions, player)

    def update(self, trajectories_and_rewards):
        keyed = []
        for traj, reward in trajectories_and_rewards:
            keyed.append(([(p, self.key_fn(s), a) for p, s, a in traj], reward))
        return super().update(keyed)


def train_ql_leduc(seed, episodes=12000):
    rng = np.random.default_rng(seed)
    env = MaskedLeducPoker(LeducPokerEnv())
    agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.15)
    for _ in range(episodes):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    return env, agent, rng


def train_masked(env, agent, rng, mask_fn, episodes):
    env.set_mask(mask_fn)
    rewards = []
    for _ in range(episodes):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
        rewards.append(r)
    return rewards


def run_public_info_ablation():
    print("\n=== Public-info vs private-info adversary (Leduc QL) ===", flush=True)
    results = {"none": [], "private_adv": [], "public_adv": [], "random": []}
    for seed in SEEDS:
        for mode in results:
            env, agent, rng = train_ql_leduc(seed)
            if mode == "none":
                results[mode].append(evaluate_agent(env, agent, rng, 2000))
            elif mode == "random":
                rm = random_mask(0, 0.5, np.random.default_rng(seed + 99))
                train_masked(env, agent, rng, rm, 8000)
                results[mode].append(evaluate_agent(env, agent, rng, 2000, mask_fn=rm))
            else:
                key_fn = (lambda x: x) if mode == "private_adv" else public_key
                adv = KeyedAdversarialMask(key_fn=key_fn, target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
                env.set_mask(adv.mask_fn)
                for _ in range(15):
                    batch = []
                    for _ in range(400):
                        r, t = play_episode(env, agent, rng)
                        agent.update(t, r)
                        batch.append((t, r))
                    adv.update(batch)
                results[mode].append(evaluate_agent(env, agent, rng, 2000, mask_fn=adv.mask_fn))
            print(f"  seed={seed} {mode} done", flush=True)
    for mode, vals in results.items():
        m, ci = mean_ci(vals)
        print(f"  {mode:>12s}: {m:+.3f} +/- {ci:.3f}", flush=True)
    return results


KUHN_STATES = ["0", "0pb", "1", "1pb", "2", "2pb"]
KUHN_REACH = {"0": 1 / 3, "1": 1 / 3, "2": 1 / 3, "0pb": 1 / 6, "1pb": 1 / 6, "2pb": 1 / 6}


def cacv_oracle_mask(agent, budget=3):
    scores = {}
    remove_action = {}
    for s in KUHN_STATES:
        q = agent.q[s]
        best = int(np.argmax(q))
        gap = float(abs(q[0] - q[1]))
        scores[s] = KUHN_REACH[s] * gap
        remove_action[s] = best
    targets = set(sorted(scores, key=scores.get, reverse=True)[:budget])

    def mask_fn(info_state, legal_actions, player):
        if player != 0 or info_state not in targets or len(legal_actions) <= 1:
            return legal_actions
        filtered = [a for a in legal_actions if a != remove_action[info_state]]
        return filtered or legal_actions

    return mask_fn, targets, scores


def run_cacv_oracle():
    print("\n=== CACv-greedy oracle (Kuhn QL, k=3) ===", flush=True)
    results = {"none": [], "random": [], "learned": [], "cacv_oracle": []}
    target_counts = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        env = MaskedKuhnPoker(KuhnPokerEnv())
        agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)
        for _ in range(10000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)

        results["none"].append(evaluate_agent(env, agent, rng, 3000))
        oracle, targets, _ = cacv_oracle_mask(agent, budget=3)
        target_counts.append(tuple(sorted(targets)))

        # Random
        rm = random_mask(0, 3 / 6, np.random.default_rng(seed + 99))
        agent_r = copy.deepcopy(agent)
        env_r = MaskedKuhnPoker(KuhnPokerEnv())
        train_masked(env_r, agent_r, rng, rm, 8000)
        results["random"].append(evaluate_agent(env_r, agent_r, rng, 3000, mask_fn=rm))

        # Learned
        agent_l = copy.deepcopy(agent)
        env_l = MaskedKuhnPoker(KuhnPokerEnv())
        adv = AdversarialMask(target_player=0, num_actions=2, lr=0.01, budget=3)
        env_l.set_mask(adv.mask_fn)
        for _ in range(20):
            batch = []
            for _ in range(300):
                r, t = play_episode(env_l, agent_l, rng)
                agent_l.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        results["learned"].append(evaluate_agent(env_l, agent_l, rng, 3000, mask_fn=adv.mask_fn))

        # Oracle
        agent_o = copy.deepcopy(agent)
        env_o = MaskedKuhnPoker(KuhnPokerEnv())
        train_masked(env_o, agent_o, rng, oracle, 8000)
        results["cacv_oracle"].append(evaluate_agent(env_o, agent_o, rng, 3000, mask_fn=oracle))
        print(f"  seed={seed} targets={sorted(targets)}", flush=True)

    for mode, vals in results.items():
        m, ci = mean_ci(vals)
        print(f"  {mode:>12s}: {m:+.3f} +/- {ci:.3f}", flush=True)
    return results, target_counts


def effective_l0(env, agent, mask_fn, rng, episodes=2000):
    masked, seen = set(), set()
    mask_events, decisions = 0, 0
    for _ in range(episodes):
        env.reset(rng=rng)
        while not env.is_terminal:
            p = env.current_player
            info = env.info_state(p)
            base = env.env.legal_actions()
            if p == 0:
                seen.add(info)
                decisions += 1
                m = mask_fn(info, base, p)
                if len(m) < len(base):
                    masked.add(info)
                    mask_events += 1
            a = agent.select_action(info, env.legal_actions(), rng)
            env.step(a)
    return len(masked), len(seen), mask_events / max(decisions, 1)


def run_l0_diagnostics():
    print("\n=== Precise L0 diagnostics (Leduc QL) ===", flush=True)
    rows = {"random": [], "fixed": [], "private_adv": [], "public_adv": []}
    for seed in SEEDS:
        env, agent, rng = train_ql_leduc(seed)
        masks = {
            "random": random_mask(0, 0.5, np.random.default_rng(seed + 99)),
            "fixed": fixed_removal(0, 2),
        }
        for name, key_fn in [("private_adv", lambda x: x), ("public_adv", public_key)]:
            adv = KeyedAdversarialMask(key_fn=key_fn, target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
            env_a, agent_a, rng_a = train_ql_leduc(seed)
            env_a.set_mask(adv.mask_fn)
            for _ in range(10):
                batch = []
                for _ in range(300):
                    r, t = play_episode(env_a, agent_a, rng_a)
                    agent_a.update(t, r)
                    batch.append((t, r))
                adv.update(batch)
            masks[name] = adv.mask_fn

        for name, mfn in masks.items():
            k, total, rate = effective_l0(MaskedLeducPoker(LeducPokerEnv()), agent, mfn, rng, 2000)
            rows[name].append((k, total, rate))
            print(f"  seed={seed} {name}: k={k}/{total} rate={rate:.3f}", flush=True)
    for name, vals in rows.items():
        vals = np.asarray(vals)
        print(f"  {name:>12s}: L0={vals[:,0].mean():.1f}, seen={vals[:,1].mean():.1f}, rate={vals[:,2].mean():.3f}", flush=True)
    return rows


def play_episode_separate(env, p0_agent, p1_agent, rng):
    env.reset(rng=rng)
    traj0, traj1 = [], []
    while not env.is_terminal:
        p = env.current_player
        info = env.info_state(p)
        legal = env.legal_actions()
        agent = p0_agent if p == 0 else p1_agent
        action = agent.select_action(info, legal, rng)
        (traj0 if p == 0 else traj1).append((0, info, action))
        env.step(action)
    return env.returns[0], traj0, traj1


def run_neural_separate_ablation():
    print("\n=== Separate-network DQN ablation (Leduc, 3 seeds) ===", flush=True)
    post_vals = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        env = MaskedLeducPoker(LeducPokerEnv())
        p0 = DQNAgent(game="leduc", num_actions=NUM_ACTIONS, lr=1e-3, epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=0.999, batch_size=32)
        p1 = DQNAgent(game="leduc", num_actions=NUM_ACTIONS, lr=1e-3, epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=0.999, batch_size=32)
        for _ in range(5000):
            r, t0, t1 = play_episode_separate(env, p0, p1, rng)
            p0.update(t0, r)
            p1.update(t1, -r)
        adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
        env.set_mask(adv.mask_fn)
        for _ in range(10):
            batch = []
            for _ in range(300):
                r, t0, t1 = play_episode_separate(env, p0, p1, rng)
                p0.update(t0, r)
                p1.update(t1, -r)
                batch.append(([(0, s, a) for _, s, a in t0], r))
            adv.update(batch)
        rewards = []
        env.set_mask(adv.mask_fn)
        for _ in range(2000):
            r, _, _ = play_episode_separate(env, p0, p1, rng)
            rewards.append(r)
        post_vals.append(float(np.mean(rewards)))
        print(f"  seed={seed}: post={post_vals[-1]:+.3f}", flush=True)
    m, ci = mean_ci(post_vals)
    print(f"  separate_dqn: {m:+.3f} +/- {ci:.3f}", flush=True)
    return post_vals


def dropout_mask(remove_prob, rng):
    return random_mask(0, remove_prob, rng)


def run_action_dropout_defense():
    print("\n=== Stochastic action-dropout defense (Leduc QL) ===", flush=True)
    results = {"standard": [], "dropout_defense": []}
    for seed in SEEDS:
        for mode in results:
            rng = np.random.default_rng(seed)
            env = MaskedLeducPoker(LeducPokerEnv())
            agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.15)
            if mode == "dropout_defense":
                dm = dropout_mask(0.2, np.random.default_rng(seed + 300))
                env.set_mask(dm)
            for _ in range(12000):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
            env.set_mask(None)
            adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
            env.set_mask(adv.mask_fn)
            for _ in range(15):
                batch = []
                for _ in range(400):
                    r, t = play_episode(env, agent, rng)
                    agent.update(t, r)
                    batch.append((t, r))
                adv.update(batch)
            results[mode].append(evaluate_agent(env, agent, rng, 2000, mask_fn=adv.mask_fn))
            print(f"  seed={seed} {mode} done", flush=True)
    for mode, vals in results.items():
        m, ci = mean_ci(vals)
        print(f"  {mode:>16s}: {m:+.3f} +/- {ci:.3f}", flush=True)
    return results


def main():
    run_public_info_ablation()
    run_cacv_oracle()
    run_l0_diagnostics()
    run_neural_separate_ablation()
    run_action_dropout_defense()


if __name__ == "__main__":
    main()
