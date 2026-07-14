import numpy as np
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg
import time

L = 100
U = 4.0
CHI_MAX = 100
SV_MIN = 1e-10

def spin_gap(L, U):
    model_params = {
        'L': L, 't': 1.0, 'U': U,
        'cons_N': 'N', 'cons_Sz': 'Sz',
        'bc_MPS': 'finite'
    }
    model = FermiHubbardChain(model_params)
    sites = model.lat.mps_sites()
    init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    
    dmrg_params = {
        'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
        'mixer': True, 'combine': True,
        'max_sweeps': 50, 'min_sweeps': 2
    }
    
    psi0 = MPS.from_product_state(sites, init_state)
    E0_result = dmrg.run(psi0, model, dmrg_params)
    E0 = E0_result['E']
    
    init_state_triplet = init_state.copy()
    center = L // 2
    init_state_triplet[center] = 'up'
    init_state_triplet[center - 1] = 'up'
    psi1 = MPS.from_product_state(sites, init_state_triplet)
    E1_result = dmrg.run(psi1, model, dmrg_params)
    E1 = E1_result['E']
    
    return E1 - E0, psi0

def compute_topology_index_C(psi, L):
    omega_sum = 0.0
    edge_count = 0
    corr_up = psi.correlation_function('Cu', 'Cdu')
    corr_down = psi.correlation_function('Cd', 'Cdd')
    for i in range(L - 1):
        bond_strength = abs(np.real(corr_up[i, i+1])) + abs(np.real(corr_down[i, i+1]))
        omega_sum += bond_strength
        edge_count += 1
    Omega = omega_sum
    N = L
    C = (Omega * edge_count) / (N * (N - 1) / 2.0)
    return C, Omega, edge_count

print(f"L={L}, U={U}")
t0 = time.time()

delta_s, psi0 = spin_gap(L, U)
C, Omega, edge_count = compute_topology_index_C(psi0, L)

delta_s_times_L = delta_s * L
elapsed = time.time() - t0

print(f"自旋能隙 Δ_s = {delta_s:.6f}")
print(f"Δ_s × L = {delta_s_times_L:.4f}")
print(f"总边权 Ω = {Omega:.6f}")
print(f"有效边数 = {edge_count}")
print(f"拓扑连通指数 C = {C:.6f}")
print(f"耗时 = {elapsed:.1f}s")
print(f"\n记录: L={L}, Δ_s={delta_s:.6f}, Δ_s×L={delta_s_times_L:.4f}, C={C:.6f}")