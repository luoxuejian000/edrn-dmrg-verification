import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

L = 40
U = 4.0
DEFECT_STRENGTH = 0.5
CHI_MAX = 100
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "periodic_defect.csv")

print("=" * 60)
print("[方向二] 周期链中的单键缺陷——拓扑关系是否调控穿透深度")
print(f"参数: L={L}, U={U}, 周期边界, 中心键缺陷={DEFECT_STRENGTH}t")
print(f"DMRG: chi_max={CHI_MAX}")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

t_array = np.ones(L)
center_bond = L // 2 - 1
t_array[center_bond] = DEFECT_STRENGTH

model_params = {
    'lattice': 'Chain',
    'L': L,
    't': t_array,
    'U': U,
    'cons_N': 'N',
    'cons_Sz': 'Sz',
    'bc_MPS': 'finite',
    'bc_x': 'periodic',
}
model = FermiHubbardModel(model_params)
sites = model.lat.mps_sites()

dmrg_params = {
    'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
    'mixer': True, 'combine': True,
    'max_sweeps': 50, 'min_sweeps': 2,
}

init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
psi0 = MPS.from_product_state(sites, init_state)
t_start = time.time()
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
for i in range(L):
    j = (i + 1) % L
    bond_up = abs(np.real(corr_up[i, j]))
    bond_dn = abs(np.real(corr_dn[i, j]))
    omega_sum += (bond_up + bond_dn) / 2.0
    edge_count += 1

C = (omega_sum * edge_count) / (L * (L - 1) / 2.0)

t_elapsed = time.time() - t_start

print(f"\n周期链 L={L}, 缺陷={DEFECT_STRENGTH}t:")
print(f"  自旋能隙 Δ_s = {delta_s:.6f}")
print(f"  前置因子 A = {A:.4f}")
print(f"  拓扑指数 C = {C:.6f}")
print(f"  耗时 = {t_elapsed:.0f}s")

with open(OUTPUT_FILE, "w") as f:
    f.write("L,boundary,defect,spin_gap,A,C,timestamp\n")
    f.write(f"{L},periodic,{DEFECT_STRENGTH},{delta_s:.10f},{A:.6f},{C:.6f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n对比数据:")
print(f"  均匀周期链:    A = 3.5262")
print(f"  均匀开链:      A = 3.0667")
print(f"  开链0.5t缺陷:  A = 5.4699 (L=40)")
print(f"  周期链0.5t缺陷: A = {A:.4f}")
print("\n判决:")
print("  若周期链0.5t缺陷的A ≈ 5.47 → 缺陷效应与拓扑无关")
print("  若周期链0.5t缺陷的A ≈ 3.53 → 周期边界压制了缺陷效应")
print("  若周期链0.5t缺陷的A是其他值 → 新的物理现象")