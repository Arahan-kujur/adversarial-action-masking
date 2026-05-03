"""Leduc-20: 20 ranks x 2 suits = 40 cards. Expecting 5000+ info states.

DQN victim + neural adversary (3-layer MLP, 128-64-4).
This is the "large" benchmark -- approaching real poker abstraction sizes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy, numpy as np, torch, torch.nn as nn, torch.optim as optim
from core.envs.leduc_n import LeducNPokerEnv, MaskedLeducNPoker, NUM_ACTIONS
from core.agents.dqn import DQNAgent, get_encoder
from core.training.selfplay import play_episode
from adversary.mask_utils import evaluate_agent
from adversary.masking_policy import random_mask, fixed_removal

NUM_RANKS = 20

class NeuralAdversary(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.num_game_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_actions + 1),
        )
        self._encoder, _ = get_encoder("leduc20")
        self._log_probs, self._rewards = [], []
        self.collecting = True

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def mask_fn(self, info_state, legal_actions, player):
        if player != 0 or len(legal_actions) <= 1:
            return legal_actions
        features = torch.FloatTensor(self._encoder(info_state)).unsqueeze(0)
        if self.collecting:
            probs = self.forward(features)[0]
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()
            self._log_probs.append(dist.log_prob(idx))
            idx = int(idx.item())
        else:
            with torch.no_grad():
                idx = int(self.forward(features)[0].argmax().item())
        if idx < self.num_game_actions and idx in legal_actions:
            f = [a for a in legal_actions if a != idx]
            if f: return f
        return legal_actions

    def record_reward(self, r):
        n = len(self._log_probs) - len(self._rewards)
        self._rewards.extend([r] * n)

    def update(self, opt):
        if not self._log_probs: return
        lp = torch.stack(self._log_probs)
        rw = torch.FloatTensor(self._rewards)
        loss = -(lp * (-(rw - rw.mean()))).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        self._log_probs.clear(); self._rewards.clear()


PRETRAIN = 30000
INNER, OUTER = 500, 25
EVAL = 3000
SEEDS = list(range(5))

def make_victim():
    return DQNAgent(game="leduc20", num_actions=NUM_ACTIONS, lr=1e-3,
                    epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99995,
                    buffer_size=50000, batch_size=128, target_update_freq=1000)

def pretrain(env, v, rng, n):
    for i in range(n):
        r, t = play_episode(env, v, rng); v.update(t, r)
        if (i+1) % 5000 == 0: print(f"      pretrain {i+1}/{n}", flush=True)

def count_states(env, agent, rng, n=30000):
    s = set()
    for _ in range(n):
        env.reset(rng=rng)
        while not env.is_terminal:
            p = env.current_player; info = env.info_state(p)
            if p == 0: s.add(info)
            a = agent.select_action(info, env.legal_actions(), rng); env.step(a)
    return s

def main():
    print("="*65, flush=True)
    print(f"  LEDUC-{NUM_RANKS}: {NUM_RANKS} ranks x 2 suits = {NUM_RANKS*2} cards", flush=True)
    print("="*65, flush=True)

    rng = np.random.default_rng(0)
    env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=NUM_RANKS))
    v = make_victim(); pretrain(env, v, rng, 5000)
    states = count_states(env, v, rng, 40000)
    print(f"\n  Unique P0 info states: {len(states)}\n", flush=True)

    results = {"None": [], "Random": [], "Fixed": [], "Neural Adv": []}
    for seed in SEEDS:
        print(f"  Seed {seed}", flush=True)
        rng = np.random.default_rng(seed); torch.manual_seed(seed)
        base = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=NUM_RANKS))
        victim = make_victim()
        print("    Pretraining...", flush=True); pretrain(base, victim, rng, PRETRAIN)

        r_none = evaluate_agent(base, victim, rng, EVAL)
        results["None"].append(r_none); print(f"    None: {r_none:+.4f}", flush=True)

        rm = random_mask(0, 0.5, np.random.default_rng(seed+99))
        r_rand = evaluate_agent(MaskedLeducNPoker(LeducNPokerEnv(num_ranks=NUM_RANKS)),
                                copy.deepcopy(victim), rng, EVAL, mask_fn=rm)
        results["Random"].append(r_rand); print(f"    Random: {r_rand:+.4f}", flush=True)

        fm = fixed_removal(0, 2)
        r_fixed = evaluate_agent(MaskedLeducNPoker(LeducNPokerEnv(num_ranks=NUM_RANKS)),
                                 copy.deepcopy(victim), rng, EVAL, mask_fn=fm)
        results["Fixed"].append(r_fixed); print(f"    Fixed: {r_fixed:+.4f}", flush=True)

        print("    Neural adversary...", flush=True)
        ne = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=NUM_RANKS))
        vn = make_victim(); pretrain(ne, vn, rng, PRETRAIN)
        _, dim = get_encoder("leduc20")
        adv = NeuralAdversary(dim, NUM_ACTIONS)
        opt = optim.Adam(adv.parameters(), lr=1e-3)
        adv.collecting = True; ne.set_mask(adv.mask_fn)
        for o in range(OUTER):
            er = []
            for _ in range(INNER):
                r, t = play_episode(ne, vn, rng); vn.update(t, r)
                adv.record_reward(r); er.append(r)
            adv.update(opt)
            if (o+1) % 5 == 0: print(f"      outer {o+1}/{OUTER}: {np.mean(er):+.4f}", flush=True)
        ne.set_mask(None); adv.collecting = False
        r_n = evaluate_agent(ne, vn, rng, EVAL, mask_fn=adv.mask_fn)
        results["Neural Adv"].append(r_n); print(f"    Neural: {r_n:+.4f}", flush=True)

    r_min, r_max = -41.0, 41.0
    print(f"\n{'='*65}\n  LEDUC-{NUM_RANKS} RESULTS ({len(SEEDS)} seeds)\n{'='*65}", flush=True)
    for s in results:
        v = results[s]; m = np.mean(v); ci = 1.96*np.std(v)/len(v)**.5
        print(f"  {s:<12s} | {m:+.3f} +/- {ci:.3f} | norm={((m-r_min)/(r_max-r_min)):.3f}", flush=True)

if __name__ == "__main__":
    main()
