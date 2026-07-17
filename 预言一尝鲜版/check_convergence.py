import time
import numpy as np
import tenpy
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

L = 200
U_OVER_T = 4.0
CHI_MAX_LIST = [100, 200, 300, 500]

print("=" * 60)
print(f"[收敛性测试] L={L}, U/t={U_OVER_T}")
print(f"[收敛性测试] chi_max values: {CHI_MAX_LIST}")
print("=" * 60)

for chi_max in CHI_MAX_LIST:
    t0 = time.time()
    model_params = {'L': L, 't': 1.0, 'U': U_OVER_T, 'cons_N': 'N', 'cons_Sz': 'Sz', 'bc_MPS': 'finite'}
    model = FermiHubbardChain(model_params)
    
    sites = model.lat.mps_sites()
    N_particles = L // 2
    N_up = N_particles // 2
    N_down = N_particles // 2
    init_state = ['up'] * N_up + ['down'] * N_down + ['empty'] * (L - N_particles)
    psi = MPS.from_product_state(
        sites, init_state, bc=model.lat.bc_MPS, unit_cell_width=model.lat.mps_unit_cell_width
    )
    
    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.0e-10,
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-10},
        'combine': True,
    }
    info = dmrg.run(psi, model, dmrg_params)
    E = info['E']
    
    elapsed = time.time() - t0
    print(f"chi_max={chi_max:4d}: E={E:.6f}, 耗时={elapsed:.0f}s")
    
    del model, psi

print("[结论] 若最后两个能量值的相对差 < 1%，则 chi_max=200 已收敛。")
print("=" * 60)