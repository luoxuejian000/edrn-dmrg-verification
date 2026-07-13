import numpy as np
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

L = 100
U = 4.0

def spin_gap(L, U):
    model_params = {
        'L': L, 't': 1.0, 'U': U,
        'cons_N': 'N', 'cons_Sz': 'Sz',
        'bc_MPS': 'finite'
    }
    model = FermiHubbardChain(model_params)
    sites = model.lat.mps_sites()
    
    init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    print(f"初始态: {init_state[:10]}... (Sz=0)")
    
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
    
    psi = MPS.from_product_state(sites, init_state)
    E0 = dmrg.run(psi, model, dmrg_params)['E']
    
    init_state_triplet = init_state.copy()
    center = L // 2
    init_state_triplet[center] = 'up'
    init_state_triplet[center - 1] = 'up'
    print(f"三重态初始态: {init_state_triplet[:10]}... (Sz=2)")
    psi_triplet = MPS.from_product_state(sites, init_state_triplet)
    E1 = dmrg.run(psi_triplet, model, dmrg_params)['E']
    
    return E1 - E0

gap = spin_gap(L, U)
print(f"L={L}, U={U}")
print(f"自旋能隙 Δ_s = {gap:.6f}")
print(f"记录: L={L}, Δ_s={gap:.6f}")