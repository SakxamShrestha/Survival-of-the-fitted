"""Compute-cost benchmark for an EMNIST-balanced-shaped distillation chain.

Measures wall-clock only. Uses random tensors of EMNIST's exact shape
(28x28x1, 47 classes) because per-step cost depends on tensor shape and
model size, not pixel values. This does NOT test whether the experiment's
dynamics work on real EMNIST.
"""
import time, torch, torch.nn as nn, torch.nn.functional as F

DEV, C, POOL, BS = "cpu", 47, 4096, 128
STEPS0, STEPS = 3000, 1500

class CNN(nn.Module):
    def __init__(self, c=C):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32*7*7,128), nn.ReLU())
        self.head = nn.Linear(128, c)
    def forward(self,x): return self.head(self.f(x))

def train(m, X, T, steps, soft, tau=1.0):
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    n = X.shape[0]
    for _ in range(steps):
        i = torch.randint(0, n, (min(BS,n),))
        out = m(X[i])
        loss = (F.kl_div(F.log_softmax(out/tau,-1), F.log_softmax(T[i]/tau,-1),
                         log_target=True, reduction="batchmean")*tau**2
                if soft else F.cross_entropy(out, T[i]))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

X = torch.randn(POOL,1,28,28)
y = torch.randint(0,C,(POOL,))
print(f"params: {sum(p.numel() for p in CNN().parameters()):,}")

t=time.time(); teacher=CNN(); train(teacher,X,y,STEPS0,False)
g0=time.time()-t; print(f"gen 0 ({STEPS0} steps): {g0:.1f}s")

for B in (4096, 128):
    times=[]
    t_local = teacher
    for g in range(3):
        with torch.no_grad(): T = t_local(X)
        sel = torch.randperm(POOL)[:B]
        s=CNN(); t0=time.time(); train(s,X[sel],T[sel],STEPS,True); times.append(time.time()-t0)
        t_local = s
    m=sum(times)/len(times)
    print(f"B={B:5d}: {m:.1f}s per generation  ({[f'{x:.1f}' for x in times]})")
    print(f"          100 gens = {m*100/60:.0f} min | 10 founders x 100 gens = {m*1000/3600:.1f} h")
