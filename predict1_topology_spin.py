import numpy as np
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg
import time
import os

L_LIST = [20, 40, 60, 80, 100]
U = 4.0
CHI_MAX = 100
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predict1_topology_spin.csv")

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
        'mixer': True,
        'combine': True,
        'max_sweeps': 50,
        'min_sweeps': 2
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
    
    delta_s = E1 - E0
    return delta_s, psi0, model

def compute_topology_index_C(psi, L):
    omega_sum = 0.0
    edge_count = 0
    
    for i in range(L - 1):
        corr_up = psi.correlation_function('Cu', 'Cdu')
        corr_down = psi.correlation_function('Cd', 'Cdd')
        bond_strength = abs(np.real(corr_up[i, i+1])) + abs(np.real(corr_down[i, i+1]))
        omega_sum += bond_strength
        edge_count += 1
    
    Omega = omega_sum
    N = L
    C = (Omega * edge_count) / (N * (N - 1) / 2.0)
    
    return C, Omega, edge_count

print("=" * 60)
print("[预言一] 自旋能隙前置因子与拓扑指数 C 的关联检验")
print(f"[审计] 时间戳: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[审计] L 序列: {L_LIST}")
print(f"[审计] U/t = {U}, chi_max = {CHI_MAX}")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    f.write("L,delta_s,delta_s_times_L,Omega,edge_count,C,timestamp\n")

for L in L_LIST:
    t_start = time.time()
    print(f"\n[计算] L={L} 开始...")
    
    delta_s, psi0, model = spin_gap(L, U)
    C, Omega, edge_count = compute_topology_index_C(psi0, L)
    
    delta_s_times_L = delta_s * L
    
    t_elapsed = time.time() - t_start
    print(f"[完成] L={L}: Δ_s={delta_s:.6f}, Δ_s×L={delta_s_times_L:.4f}, Ω={Omega:.6f}, C={C:.6f}, 耗时={t_elapsed:.1f}s")
    
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"{L},{delta_s:.10f},{delta_s_times_L:.6f},{Omega:.10f},{edge_count},{C:.10f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n" + "=" * 60)
print("[预言一] 检验数据收集完成")
print(f"[审计] 数据已保存至: {os.path.abspath(OUTPUT_FILE)}")
print("=" * 60)
print("\n[分析提示]")
print("若 Δ_s × L 与 C 存在关联，则预言一获得初步支持。")
print("若不存在关联，则预言一需要修正或放弃。")
print("无论结果如何，数据本身是诚实的——只看病，不开方。")