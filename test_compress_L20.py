import numpy as np
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

L = 20
U = 4.0

def ground_state_energy(L, Np):
    model_params = {
        'L': L, 't': 1.0, 'U': U,
        'cons_N': 'N', 'cons_Sz': None,
        'bc_MPS': 'finite'
    }
    model = FermiHubbardChain(model_params)
    
    sites = model.lat.mps_sites()
    init_state = ['empty'] * L
    
    if Np <= L:
        for i in range(Np):
            init_state[i] = 'up' if i % 2 == 0 else 'down'
    else:
        for i in range(L):
            init_state[i] = 'up'
        N_extra = Np - L
        for i in range(N_extra):
            init_state[i] = 'full'
    
    psi = MPS.from_product_state(sites, init_state)
    
    dmrg_params = {
        'trunc_params': {
            'chi_max': 200,
            'svd_min': 1.e-10
        },
        'mixer': True,
        'combine': True
    }
    eng = dmrg.run(psi, model, dmrg_params)
    return eng['E']

print(f"计算 L={L}，U={U}")
E_minus = ground_state_energy(L, L-1)
print(f"粒子数={L-1}, E={E_minus:.6f}")
E_half = ground_state_energy(L, L)
print(f"粒子数={L}, E={E_half:.6f}")
E_plus = ground_state_energy(L, L+1)
print(f"粒子数={L+1}, E={E_plus:.6f}")

compressibility = 2.0 / (E_plus + E_minus - 2*E_half)
print(f"\n压缩率 χ = {compressibility:.6f}")
print("如果 χ 为非零正数，则测量成功。")