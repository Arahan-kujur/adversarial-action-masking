"""Generate the unified comparison table for the paper."""
import sys; sys.path.insert(0, ".")
import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker
from core.agents.q_learning import QLearningAgent
from core.agents.ppo import TabularPPOAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask, fixed_removal
from adversary.mask_utils import evaluate_agent

seeds = [42, 123, 456]

GAMES = {
    "Kuhn": (KuhnPokerEnv, MaskedKuhnPoker, 2, -2, 2),
    "Leduc": (LeducPokerEnv, MaskedLeducPoker, 3, -13, 13),
}
AGENTS = {
    "QL": (QLearningAgent, lambda na: {"num_actions": na, "alpha": 0.1, "epsilon": 0.15}),
    "PPO": (TabularPPOAgent, lambda na: {"num_actions": na, "lr": 0.01}),
}

def norm(r, mn, mx):
    return (r - mn) / (mx - mn)

def run(env_cls, masked_cls, na, agent_cls, kwargs, mask_mode, seed):
    rng = np.random.default_rng(seed)
    env = masked_cls(env_cls())
    agent = agent_cls(**kwargs)
    for _ in range(10000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    if mask_mode == "none":
        return evaluate_agent(env, agent, rng, 3000)
    elif mask_mode == "random":
        m = random_mask(0, 0.5, np.random.default_rng(seed + 99))
        env.set_mask(m)
        for _ in range(10000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        return evaluate_agent(env, agent, rng, 3000, mask_fn=m)
    elif mask_mode == "fixed":
        m = fixed_removal(0, na - 1)
        env.set_mask(m)
        for _ in range(10000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        return evaluate_agent(env, agent, rng, 3000, mask_fn=m)
    elif mask_mode == "adversarial":
        adv = AdversarialMask(target_player=0, num_actions=na, lr=0.01)
        env.set_mask(adv.mask_fn)
        for o in range(20):
            batch = []
            for _ in range(500):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        return evaluate_agent(env, agent, rng, 3000, mask_fn=adv.mask_fn)

print("Unified Comparison Table (3 seeds)")
print()
print(f"{'Setting':<15s}  {'None':>8s}  {'Random':>8s}  {'Fixed':>8s}  {'Advers.':>8s}  |  {'None(n)':>8s}  {'Rand(n)':>8s}  {'Fix(n)':>8s}  {'Adv(n)':>8s}")
print("-" * 100)

for gname, (ec, mc, na, mn, mx) in GAMES.items():
    for aname, (ac, kfn) in AGENTS.items():
        kw = kfn(na)
        vals = {}
        for mode in ["none", "random", "fixed", "adversarial"]:
            rs = [run(ec, mc, na, ac, kw, mode, s) for s in seeds]
            vals[mode] = np.mean(rs)
        nv = {k: norm(v, mn, mx) for k, v in vals.items()}
        label = f"{gname} + {aname}"
        print(f"{label:<15s}  {vals['none']:>+8.3f}  {vals['random']:>+8.3f}  "
              f"{vals['fixed']:>+8.3f}  {vals['adversarial']:>+8.3f}  |  "
              f"{nv['none']:>8.3f}  {nv['random']:>8.3f}  {nv['fixed']:>8.3f}  {nv['adversarial']:>8.3f}")
