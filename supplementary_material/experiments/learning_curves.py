"""Learning curves under adversarial masking for DQN and NFSP in Leduc.

Shows whether victims converge (asymptotic) or are still transient.
Reports reward in windows of 500 episodes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.agents.nfsp import NFSPAgent
from core.agents.dqn import DQNAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask

WINDOW = 500


def run_learning_curve(game, agent_type, seed):
    rng = np.random.default_rng(seed)

    if game == "kuhn":
        env = MaskedKuhnPoker(KuhnPokerEnv())
        n_actions = 2
        pretrain = 10000
        attack_eps = 15000
    else:
        env = MaskedLeducPoker(LeducPokerEnv())
        n_actions = NUM_ACTIONS
        pretrain = 15000
        attack_eps = 15000

    if agent_type == "ql":
        agent = QLearningAgent(num_actions=n_actions, alpha=0.1, epsilon=0.15)
    elif agent_type == "nfsp":
        agent = NFSPAgent(num_actions=n_actions, alpha=0.1, epsilon=0.15, eta=0.1)
    elif agent_type == "dqn":
        import torch
        torch.manual_seed(seed)
        agent = DQNAgent(game=game, num_actions=n_actions, lr=1e-3,
                         epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=0.999,
                         buffer_size=10000, batch_size=32, target_update_freq=200)

    # Pretrain
    pre_rewards = []
    for i in range(pretrain):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
        pre_rewards.append(r)

    # Attack
    adv = AdversarialMask(target_player=0, num_actions=n_actions, lr=0.01)
    env.set_mask(adv.mask_fn)
    post_rewards = []
    for i in range(attack_eps):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
        post_rewards.append(r)
        if (i + 1) % WINDOW == 0:
            batch = [(t, r)]
            adv.update(batch)

    return pre_rewards, post_rewards


def summarize(rewards, window=WINDOW):
    """Return list of (episode_center, mean_reward) tuples."""
    pts = []
    for i in range(0, len(rewards), window):
        chunk = rewards[i:i+window]
        if chunk:
            pts.append((i + len(chunk)//2, np.mean(chunk)))
    return pts


def main():
    configs = [
        ("kuhn", "ql"), ("kuhn", "nfsp"),
        ("leduc", "ql"), ("leduc", "nfsp"),
    ]

    for game, agent_type in configs:
        print(f"\n{'='*60}", flush=True)
        print(f"  Learning Curve: {game} + {agent_type} (seed=42)", flush=True)
        print(f"{'='*60}", flush=True)

        pre, post = run_learning_curve(game, agent_type, seed=42)

        print(f"  PRE-ATTACK (last 5 windows):", flush=True)
        for ep, mr in summarize(pre)[-5:]:
            print(f"    ep={ep:>6d}: {mr:+.4f}", flush=True)

        print(f"  POST-ATTACK (all windows):", flush=True)
        for ep, mr in summarize(post):
            print(f"    ep={ep:>6d}: {mr:+.4f}", flush=True)

        final_pre = np.mean(pre[-WINDOW:])
        final_post = np.mean(post[-WINDOW:])
        print(f"  SUMMARY: pre_final={final_pre:+.4f} -> post_final={final_post:+.4f}"
              f"  delta={final_post-final_pre:+.4f}", flush=True)


if __name__ == "__main__":
    main()
