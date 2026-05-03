"""Neural NFSP on Leduc-5: proper neural average policy under adversarial masking."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np, torch
from core.envs.leduc_n import LeducNPokerEnv, MaskedLeducNPoker, NUM_ACTIONS
from core.agents.neural_nfsp import NeuralNFSPAgent
from core.agents.dqn import get_encoder
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask
from adversary.mask_utils import evaluate_agent

SEEDS = list(range(5))
PRETRAIN = 20000
ADV_OUTER, ADV_INNER = 25, 500
EVAL = 3000


def run_one(seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))
    encoder, dim = get_encoder("leduc5")
    agent = NeuralNFSPAgent(encoder, dim, NUM_ACTIONS, eta=0.1,
                            br_lr=1e-3, avg_lr=1e-3, epsilon=0.15,
                            buffer_size=20000, avg_buffer_size=50000, batch_size=64)
    for i in range(PRETRAIN):
        r, t = play_episode(env, agent, rng); agent.update(t, r)
        if (i+1) % 5000 == 0: print(f"    pretrain {i+1}/{PRETRAIN}", flush=True)

    pre = evaluate_agent(env, agent, rng, EVAL)
    print(f"    pre-attack: {pre:+.4f}", flush=True)

    # Adversarial
    adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
    env.set_mask(adv.mask_fn)
    for o in range(ADV_OUTER):
        batch = []
        for _ in range(ADV_INNER):
            r, t = play_episode(env, agent, rng); agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)
        if (o+1) % 5 == 0:
            print(f"    attack outer {o+1}/{ADV_OUTER}", flush=True)

    post_adv = evaluate_agent(env, agent, rng, EVAL, mask_fn=adv.mask_fn)

    # Random baseline (fresh agent)
    env2 = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))
    agent2 = NeuralNFSPAgent(encoder, dim, NUM_ACTIONS, eta=0.1)
    rng2 = np.random.default_rng(seed + 100)
    for _ in range(PRETRAIN):
        r, t = play_episode(env2, agent2, rng2); agent2.update(t, r)
    rm = random_mask(0, 0.5, np.random.default_rng(seed + 99))
    env2.set_mask(rm)
    for _ in range(ADV_OUTER * ADV_INNER):
        r, t = play_episode(env2, agent2, rng2); agent2.update(t, r)
    post_rand = evaluate_agent(env2, agent2, rng2, EVAL, mask_fn=rm)

    print(f"    post-adv: {post_adv:+.4f}  post-rand: {post_rand:+.4f}", flush=True)
    return pre, post_adv, post_rand


def main():
    print("="*65, flush=True)
    print("  Neural NFSP on Leduc-5 (128-64 MLP avg policy)", flush=True)
    print("="*65, flush=True)

    pre_all, adv_all, rand_all = [], [], []
    for seed in SEEDS:
        print(f"\n  Seed {seed}:", flush=True)
        pre, adv, rand = run_one(seed)
        pre_all.append(pre); adv_all.append(adv); rand_all.append(rand)

    def fmt(v): return f"{np.mean(v):+.3f} +/- {1.96*np.std(v)/len(v)**.5:.3f}"
    print(f"\n{'='*65}\n  NEURAL NFSP LEDUC-5 RESULTS\n{'='*65}", flush=True)
    print(f"  Pre:       {fmt(pre_all)}", flush=True)
    print(f"  Post-adv:  {fmt(adv_all)}", flush=True)
    print(f"  Post-rand: {fmt(rand_all)}", flush=True)
    da = np.mean(adv_all) - np.mean(pre_all)
    dr = np.mean(rand_all) - np.mean(pre_all)
    print(f"  Delta(adv): {da:+.3f}  Delta(rand): {dr:+.3f}", flush=True)
    if dr != 0: print(f"  Ratio: {da/dr:.1f}x", flush=True)

if __name__ == "__main__":
    main()
