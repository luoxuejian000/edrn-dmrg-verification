import numpy as np
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

L = 200
U = 4.0

def ground_state_energy(L, Np):
    model_params = {
        'L': L, 't': 1.0, 'U': U,
        'cons_N': 'N', 'cons_Sz': None,
        'bc_MPS': 'finite'
    }
    model = FermiHubbardChain(model_params)
    sites = model.lat.mps_sites()

    if Np <= L:
        init_state = [('up' if i % 2 == 0 else 'down') for i in range(Np)]
        init_state += ['empty'] * (L - Np)
    else:
        n_full = Np - L
        n_single = L - n_full
        init_state = []
        for i in range(n_full):
            init_state.append('full')
        for i in range(n_single):
            init_state.append('up' if i % 2 == 0 else 'down')

    psi = MPS.from_product_state(sites, init_state)

    dmrg_params = {
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-10
        },
        'mixer': True,
        'combine': True,
        'max_sweeps': 50,
        'min_sweeps': 2
    }
    eng = dmrg.run(psi, model, dmrg_params)
    return eng['E']

print(f"L={L}, U={U}")
E_minus = ground_state_energy(L, L - 1)
print(f"  粒子数={L-1}, E={E_minus:.6f}")
E_half  = ground_state_energy(L, L)
print(f"  粒子数={L}, E={E_half:.6f}")

gap = E_half - E_minus
chi = 1.0 / gap
print(f"\n能隙 = {gap:.6f}")
print(f"压缩率 chi = {chi:.6f}")
print(f"记录: L={L}, chi={chi:.6f}")