"""Gap vs game size: adversarial advantage grows with complexity.

Runs adversarial and random masking across Kuhn, Leduc, Leduc-5.
Reports the gap (adv - random) at each scale.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask
from adversary.mask_utils import evaluate_agent

SEEDS = list(range(5))


def run_game(game_name, seed):
    rng = np.random.default_rng(seed)

    if game_name == "kuhn":
        from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
        env = MaskedKuhnPoker(KuhnPokerEnv())
        n_actions = 2
        pretrain = 10000
    elif game_name == "leduc":
        from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker
        env = MaskedLeducPoker(LeducPokerEnv())
        n_actions = 3
        pretrain = 15000
    elif game_name == "leduc5":
        from core.envs.leduc_n import LeducNPokerEnv, MaskedLeducNPoker
        env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))
        n_actions = 3
        pretrain = 20000

    agent = QLearningAgent(num_actions=n_actions, alpha=0.1, epsilon=0.15)
    for _ in range(pretrain):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    none_r = evaluate_agent(env, agent, rng, 3000)

    # Adversarial
    adv = AdversarialMask(target_player=0, num_actions=n_actions, lr=0.01)
    env.set_mask(adv.mask_fn)
    for _ in range(20):
        batch = []
        for _ in range(500):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)
    adv_r = evaluate_agent(env, agent, rng, 3000, mask_fn=adv.mask_fn)

    # Random
    env2_rng = np.random.default_rng(seed + 200)
    if game_name == "kuhn":
        from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
        env2 = MaskedKuhnPoker(KuhnPokerEnv())
    elif game_name == "leduc":
        from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker
        env2 = MaskedLeducPoker(LeducPokerEnv())
    elif game_name == "leduc5":
        from core.envs.leduc_n import LeducNPokerEnv, MaskedLeducNPoker
        env2 = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))

    agent2 = QLearningAgent(num_actions=n_actions, alpha=0.1, epsilon=0.15)
    for _ in range(pretrain):
        r, t = play_episode(env2, agent2, env2_rng)
        agent2.update(t, r)
    rm = random_mask(0, 0.5, np.random.default_rng(seed + 99))
    env2.set_mask(rm)
    for _ in range(10000):
        r, t = play_episode(env2, agent2, env2_rng)
        agent2.update(t, r)
    rand_r = evaluate_agent(env2, agent2, env2_rng, 3000, mask_fn=rm)

    return none_r, adv_r, rand_r


def main():
    games = [
        ("kuhn", "Kuhn (6 states)"),
        ("leduc", "Leduc (~50 states)"),
        ("leduc5", "Leduc-5 (389 states)"),
    ]

    print("Gap vs Game Size (5 seeds)", flush=True)
    print(f"{'Game':>25s}  {'None':>8s}  {'Advers.':>8s}  {'Random':>8s}  "
          f"{'Gap':>8s}  {'Ratio':>6s}", flush=True)
    print("-" * 75, flush=True)

    for game_id, game_name in games:
        none_vals, adv_vals, rand_vals = [], [], []
        for seed in SEEDS:
            n, a, r = run_game(game_id, seed)
            none_vals.append(n)
            adv_vals.append(a)
            rand_vals.append(r)
            print(f"    {game_id} seed={seed} done", flush=True)

        mn = np.mean(none_vals)
        ma = np.mean(adv_vals)
        mr = np.mean(rand_vals)
        gap = ma - mr
        ratio = (mn - ma) / max(abs(mn - mr), 0.001)
        print(f"{game_name:>25s}  {mn:+.3f}  {ma:+.3f}  {mr:+.3f}  "
              f"{gap:+.3f}  {ratio:.1f}x", flush=True)

    print(flush=True)


if __name__ == "__main__":
    main()
