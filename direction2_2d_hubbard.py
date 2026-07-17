import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

Lx = 2
Ly = 4
U = 4.0
CHI_MAX = 100
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "direction2_2d_hubbard.csv")

print("=" * 60)
print("[方向二] 沉默失谐跨系统验证——二维Hubbard模型")
print(f"参数: Lx={Lx}, Ly={Ly}, U={U}")
print(f"DMRG: chi_max={CHI_MAX}")
print("=" * 60)
print("[方法论声明]")
print("  本实验仅检验 α_charge + α_spin = -1 是否成立。")
print("  不预设结论，不选择性报告，只让数据说话。")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

model_params = {
    'lattice': 'Square',
    'Lx': Lx,
    'Ly': Ly,
    't': 1.0,
    'U': U,
    'bc_MPS': 'finite',
    'bc_x': 'open',
    'bc_y': 'open',
    'conserve': 'best',
    'verbose': False,
}

params_charge = model_params.copy()
params_charge['conserve'] = 'N'
model_charge = FermiHubbardModel(params_charge)
sites_charge = model_charge.lat.mps_sites()
L_sites = len(sites_charge)

dmrg_params = {
    'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
    'mixer': True, 'combine': True,
    'max_sweeps': 30, 'min_sweeps': 2,
}

half_filling = L_sites
init_state_half = ['up' if i % 2 == 0 else 'down' for i in range(half_filling)]
init_state_half += ['empty'] * (L_sites - half_filling)
psi_half = MPS.from_product_state(sites_charge, init_state_half)
E_half = dmrg.run(psi_half, model_charge, dmrg_params)['E']

init_state_minus = ['up' if i % 2 == 0 else 'down' for i in range(half_filling - 1)]
init_state_minus += ['empty'] * (L_sites - half_filling + 1)
psi_minus = MPS.from_product_state(sites_charge, init_state_minus)
E_minus = dmrg.run(psi_minus, model_charge, dmrg_params)['E']

charge_gap = E_half - E_minus

params_spin = model_params.copy()
params_spin['conserve'] = 'N,Sz'
model_spin = FermiHubbardModel(params_spin)
sites_spin = model_spin.lat.mps_sites()

init_state_singlet = ['up' if i % 2 == 0 else 'down' for i in range(half_filling)]
init_state_singlet += ['empty'] * (L_sites - half_filling)
psi_singlet = MPS.from_product_state(sites_spin, init_state_singlet)
E_singlet = dmrg.run(psi_singlet, model_spin, dmrg_params)['E']

init_state_triplet = init_state_singlet.copy()
center = L_sites // 2
if init_state_triplet[center] == 'down':
    init_state_triplet[center] = 'up'
else:
    init_state_triplet[center] = 'up'
    init_state_triplet[center - 1] = 'up'
psi_triplet = MPS.from_product_state(sites_spin, init_state_triplet)
E_triplet = dmrg.run(psi_triplet, model_spin, dmrg_params)['E']

spin_gap = E_triplet - E_singlet

L_eff = np.sqrt(L_sites)
log_L = np.log(L_eff)

print(f"\n二维Hubbard (Lx={Lx}, Ly={Ly}, U={U}):")
print(f"  格点数 = {L_sites}")
print(f"  电荷能隙 ΔE = {charge_gap:.6f}")
print(f"  自旋能隙 Δ_s = {spin_gap:.6f}")
print(f"  有效尺寸 L_eff = {L_eff:.2f}")
print(f"  log(L_eff) = {log_L:.4f}")
print(f"  log(ΔE) = {np.log(charge_gap):.4f}")
print(f"  log(Δ_s) = {np.log(spin_gap):.4f}")

with open(OUTPUT_FILE, "w") as f:
    f.write("system,Lx,Ly,U,charge_gap,spin_gap,log_L_eff,log_charge_gap,log_spin_gap,timestamp\n")
    f.write(f"2d_hubbard,{Lx},{Ly},{U},{charge_gap:.10f},{spin_gap:.10f},{log_L:.6f},{np.log(charge_gap):.6f},{np.log(spin_gap):.6f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n" + "=" * 60)
print("[方向二] 第一步完成")
print("=" * 60)
print("\n接下来需要至少再跑一个尺寸（如 2x6 或 3x4）来提取标度指数。")
print("然后计算 α_charge + α_spin 是否等于 -1。")
print("本实验仅诊断，不做规范性判断。")