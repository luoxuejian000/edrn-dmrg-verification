import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

L = 40
U = 4.0
DEFECT_LIST = [0.5, 1.0, 1.2, 1.5]
CHI_MAX = 100
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "game1_log.csv")

print("=" * 60)
print("[游戏一·批量版] 预言一尝鲜 - 单键缺陷多点扫描")
print(f"参数: L={L}, U={U}")
print(f"缺陷强度序列: {DEFECT_LIST}")
print(f"DMRG: chi_max={CHI_MAX}")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    f.write("defect_strength,delta_s,A,C,timestamp\n")

for ds in DEFECT_LIST:
    t_start = time.time()
    print(f"\n[运行] 缺陷强度 = {ds}t ...")

    t_array = np.ones(L - 1)
    center_bond = L // 2 - 1
    t_array[center_bond] = ds

    model_params = {
        'L': L, 't': t_array, 'U': U,
        'cons_N': 'N', 'cons_Sz': 'Sz',
        'bc_MPS': 'finite',
    }
    model = FermiHubbardChain(model_params)
    sites = model.lat.mps_sites()

    dmrg_params = {
        'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
        'mixer': True, 'combine': True,
        'max_sweeps': 50, 'min_sweeps': 2,
    }

    init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    psi0 = MPS.from_product_state(sites, init_state)
    E0 = dmrg.run(psi0, model, dmrg_params)['E']

    init_state_triplet = init_state.copy()
    center = L // 2
    init_state_triplet[center] = 'up'
    init_state_triplet[center - 1] = 'up'
    psi1 = MPS.from_product_state(sites, init_state_triplet)
    E1 = dmrg.run(psi1, model, dmrg_params)['E']

    delta_s = E1 - E0
    A = delta_s * L

    corr_up = psi0.correlation_function('Cdu', 'Cu')
    corr_dn = psi0.correlation_function('Cdd', 'Cd')

    omega_sum = 0.0
    edge_count = 0
    for i in range(L - 1):
        bond_up = abs(np.real(corr_up[i, i+1]))
        bond_dn = abs(np.real(corr_dn[i, i+1]))
        omega_sum += (bond_up + bond_dn) / 2.0
        edge_count += 1

    C = (omega_sum * edge_count) / (L * (L - 1) / 2.0)

    t_elapsed = time.time() - t_start
    print(f"  缺陷={ds}t: Δ_s={delta_s:.6f}, A={A:.4f}, C={C:.6f}, 耗时={t_elapsed:.0f}s")

    with open(OUTPUT_FILE, "a") as f:
        f.write(f"{ds},{delta_s:.10f},{A:.6f},{C:.6f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n" + "=" * 60)
print("[游戏一·批量版] 全部完成")
print(f"数据已保存至: {os.path.abspath(OUTPUT_FILE)}")
print("=" * 60)
print("\n待收集的数据点:")
print("  均匀开链（基准）: A≈3.067, C≈0.959")
print("  缺陷 0.8t:        A=4.367,  C=0.478")
print("  缺陷 0.5t:        ?")
print("  缺陷 1.0t:        ?")
print("  缺陷 1.2t:        ?")
print("  缺陷 1.5t:        ?")
print("\n画散点图 (C vs A)，看是否单调。")