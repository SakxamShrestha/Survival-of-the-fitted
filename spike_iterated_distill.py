"""
Minimal spike for "Surviving the Bottleneck": iterated knowledge distillation
across generations, seeded either from memorized random labels (noise) or from a
compositional labeling function (structured).

Run this yourself before you build on the pilot numbers in the proposal.

    python3 spike_iterated_distill.py --gens 40 --tau 1.0 --bottleneck 4096
    python3 spike_iterated_distill.py --gens 40 --tau 2.0
    python3 spike_iterated_distill.py --gens 40 --tau 0.5
    python3 spike_iterated_distill.py --gens 40 --mode argmax
    python3 spike_iterated_distill.py --gens 40 --bottleneck 128 # tight bottleneck
    python3 spike_iterated_distill.py --gens 40 --seed-kind structured

Two warnings about what these runs do and do not show.

**Temperature does not order the decay.** Earlier versions of this docstring labelled
`--tau 0.5` a "frozen fixed point" and `--tau 2.0` as dissolving fast. Neither reproduced.
Control C-6 then showed the apparent ordering exists only on the pool the chain trains on:
measured on held-out inputs the three temperatures land within 0.013 of each other, which
is a smaller spread than chance agreement (0.020), and not in order.

**`--mode argmax` is not an absorbing state, and its fidelity is mostly memorization.**
It reaches 0.654 agreement with generation 0 at generation 25 on the training pool but
0.167 on held-out inputs — a gap of 0.753 already at generation 1, before any chaining.
It preserves the founder's memorized answers on the points it memorized, not its function.

Metrics printed here are computed on the training pool, which at `--bottleneck 4096` means
the students train on every point being measured. For held-out measurement use
`controls/c6_probe_pool.py`.
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- data

def make_pool(n=4096, k=5, v=10, seed=0, kind="noise"):
    """Fixed pool of one-hot attribute vectors plus generation-0 labels."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v, size=(n, k))
    X = np.zeros((n, k * v), dtype=np.float32)
    X[np.arange(n)[:, None], np.arange(k) * v + idx] = 1.0
    n_classes = k * v
    if kind == "structured":
        y = (idx[:, 0] * v + idx[:, 1]) % n_classes   # compositional in 2 attributes
    else:
        y = rng.integers(0, n_classes, size=n)        # pure random labels
    return torch.from_numpy(X), torch.from_numpy(y).long(), n_classes


# ---------------------------------------------------------------- model

class MLP(nn.Module):
    def __init__(self, d_in, d_out, h=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.head = nn.Linear(h, d_out)

    def forward(self, x, return_feats=False):
        z = self.body(x)
        logits = self.head(z)
        return (logits, z) if return_feats else logits


# ---------------------------------------------------------------- metrics

def effective_rank(Z):
    """exp(-sum p log p) over the normalized eigenvalue spectrum of cov(Z)."""
    Z = Z - Z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Z)
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    p = p.clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def linear_cka(X, Y):
    """Kornblith et al. 2019, linear CKA on column-centered features."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = (X.T @ Y).norm() ** 2
    xtx = (X.T @ X).norm()
    yty = (Y.T @ Y).norm()
    return float(xty / (xtx * yty).clamp_min(1e-12))


# ---------------------------------------------------------------- training

def train(model, X, targets, steps, bs, lr, wd, soft, tau, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = X.shape[0]
    model.train()
    for _ in range(steps):
        i = torch.randint(0, n, (min(bs, n),), device=device)
        logits = model(X[i])
        if soft:
            # KL(teacher || student) at temperature tau, standard tau^2 scaling
            loss = F.kl_div(
                F.log_softmax(logits / tau, dim=-1),
                F.log_softmax(targets[i] / tau, dim=-1),
                log_target=True, reduction="batchmean",
            ) * (tau ** 2)
        else:
            loss = F.cross_entropy(logits, targets[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--bottleneck", type=int, default=4096)
    ap.add_argument("--mode", choices=["soft", "sampled", "argmax"], default="soft")
    ap.add_argument("--seed-kind", choices=["noise", "structured"], default="noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps0", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--log", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"
    X, y0, C = make_pool(seed=args.seed, kind=args.seed_kind)
    X, y0 = X.to(device), y0.to(device)

    # generation 0: memorize its assigned labels
    t0 = time.time()
    teacher = MLP(X.shape[1], C).to(device)
    train(teacher, X, y0, args.steps0, 256, 1e-3, args.wd, False, 1.0, device)

    with torch.no_grad():
        base_logits, base_feats = teacher(X, return_feats=True)
    base_pred = base_logits.argmax(-1)
    print(f"gen 0 memorization accuracy on its own labels: "
          f"{(base_pred == y0).float().mean():.3f}  ({time.time()-t0:.1f}s)")

    prev_pred, prev_feats = base_pred, base_feats
    rows = []
    for g in range(1, args.gens + 1):
        with torch.no_grad():
            t_logits = teacher(X)
        sel = torch.randperm(X.shape[0])[: args.bottleneck]
        Xs, Ts = X[sel], t_logits[sel]

        student = MLP(X.shape[1], C).to(device)
        if args.mode == "soft":
            train(student, Xs, Ts, args.steps, 256, 1e-3, args.wd, True, args.tau, device)
        else:
            if args.mode == "argmax":
                hard = Ts.argmax(-1)
            else:  # sampled
                hard = torch.multinomial(F.softmax(Ts / args.tau, -1), 1).squeeze(-1)
            train(student, Xs, hard, args.steps, 256, 1e-3, args.wd, False, 1.0, device)

        with torch.no_grad():
            logits, feats = student(X, return_feats=True)
        pred = logits.argmax(-1)
        probs = F.softmax(logits, -1)

        row = dict(
            gen=g,
            agree_gen0=float((pred == base_pred).float().mean()),
            agree_prev=float((pred == prev_pred).float().mean()),
            entropy=float(-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()),
            logit_norm=float(logits.norm(dim=-1).mean()),
            n_classes_used=int(pred.unique().numel()),
            eff_rank=effective_rank(feats),
            cka_gen0=linear_cka(feats, base_feats),
            cka_prev=linear_cka(feats, prev_feats),
        )
        rows.append(row)
        if g % 5 == 0 or g == 1:
            print(f"g{g:3d}  agree0={row['agree_gen0']:.3f}  agreeprev={row['agree_prev']:.3f}  "
                  f"H={row['entropy']:.3f}  |z|={row['logit_norm']:6.2f}  "
                  f"classes={row['n_classes_used']:3d}  effrank={row['eff_rank']:6.2f}  "
                  f"CKA0={row['cka_gen0']:.3f}  CKAprev={row['cka_prev']:.3f}")

        teacher, prev_pred, prev_feats = student, pred, feats

    print(f"chance agreement = {1.0/C:.3f}   max entropy = {np.log(C):.3f}   "
          f"total {time.time()-t0:.1f}s")
    if args.log:
        with open(args.log, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
