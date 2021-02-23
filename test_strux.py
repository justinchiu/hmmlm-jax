import torch
import torch_struct

import numpy as np
import jax
import strux

import importlib.util
spec = importlib.util.spec_from_file_location(
    "get_fb",
    "hmm_runners/hmm.py",
)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

N, T, H = 4, 16, 32
fb = foo.get_fb(H)

log_pots = np.random.rand(N, T, H, H).astype(np.float32)
log_pots_t = torch.tensor(log_pots)
lengths = np.empty(N)
lengths.fill(T+1)

lc_t = torch_struct.LinearChain()
lc_j = strux.LinearChain()

log_M, alphas = fb(log_pots_t.cuda())

Zj = lc_j.sum(log_pots, lengths)
Zt = lc_t.sum(log_pots_t)
Zf = alphas[-1].logsumexp(-1)
print(Zj, Zt, Zf)

Mj = lc_j.marginals(log_pots, lengths)
Mt = lc_t.marginals(log_pots_t)
Mf = log_M.exp()
print(np.abs(Mj - Mt.cpu().detach().numpy()).max())
print(np.abs(Mj - Mf.cpu().detach().numpy()).max())

with open("test.npy", "rb") as f:
    lpnp = np.load(f)
    #lpnp = lpnp.transpose((0,1,3,2))

N, T, H, _ = lpnp.shape
fb = foo.get_fb(H)

lengths = np.empty(N)
lengths.fill(T+1)
lp = torch.tensor(lpnp).cuda()

log_M, alphas = fb(lp)

Zj = lc_j.sum(lpnp, lengths)
#Zt = lc_t.sum(lp)
Zf = alphas[-1].logsumexp(-1)
print(Zj, Zt, Zf)
print(Zj.sum(), Zt.sum().item(), Zf.sum().item())

#Mj = lc_j.marginals(lpnp, lengths)
#Mt = lc_t.marginals(lp)
#Ej = (Mj * lpnp)
#Ej = Ej[Ej == Ej].sum()
#Et = (Mt * lp)
#Et = Et[Et == Et].sum()
Ej = 0
Et = 0
Ef = (log_M.exp() * lp)
Ef = Ef[Ef == Ef].sum()
print(Ej, Et, Ef)

import pdb; pdb.set_trace()
