import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

L = 40
U = 4.0
CHI_MAX = 200
SV_MIN = 1e-10
MAX_SWEEPS = 100
MIN_SWEEPS = 2
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "menu2_periodic.csv")

print("=" * 60)
print("[菜单二] 周期链对比实验——检验“关系先于实体”")
print(f"参数: L={L}, U={U}, 周期边界")
print(f"DMRG: chi_max={CHI_MAX}, max_sweeps={MAX_SWEEPS}")
print("=" * 60)
print("[方法论声明]")
print("  1. 所有局域参数与均匀开链完全相同（t=1.0t）。")
print("  2. 唯一改变的是边界条件——纯粹的拓扑变化。")
print("  3. 不预设A是否变化——数据说话。")
print("  4. 若数据不支持，诚实报告。")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

model_params = {
    'lattice': 'Chain',
    'L': L,
    't': 1.0,
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
    'mixer': True,
    'mixer_params': {'amplitude': 1e-4, 'decay': 1.5},
    'combine': True,
    'max_sweeps': MAX_SWEEPS,
    'min_sweeps': MIN_SWEEPS,
}

init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
psi0 = MPS.from_product_state(sites, init_state)
t0 = time.time()
try:
    E0 = dmrg.run(psi0, model, dmrg_params)['E']
except Exception as e:
    print(f"[错误] 单态DMRG失败: {e}")
    print("记录哨兵值-999")
    E0 = -999.0

if E0 > -900:
    init_state_triplet = init_state.copy()
    center = L // 2
    init_state_triplet[center] = 'up'
    init_state_triplet[center - 1] = 'up'
    psi1 = MPS.from_product_state(sites, init_state_triplet)
    try:
        E1 = dmrg.run(psi1, model, dmrg_params)['E']
    except Exception as e:
        print(f"[错误] 三重态DMRG失败: {e}")
        print("记录哨兵值-999")
        E1 = -999.0

    if E1 > -900:
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

        model_params_charge = {
            'lattice': 'Chain',
            'L': L,
            't': 1.0,
            'U': U,
            'cons_N': 'N',
            'cons_Sz': None,
            'bc_MPS': 'finite',
            'bc_x': 'periodic',
        }
        model_charge = FermiHubbardModel(model_params_charge)
        sites_charge = model_charge.lat.mps_sites()
        init_state_minus = [('up' if i % 2 == 0 else 'down') for i in range(L-1)] + ['empty']
        psi_minus = MPS.from_product_state(sites_charge, init_state_minus)
        try:
            E_minus = dmrg.run(psi_minus, model_charge, dmrg_params)['E']
        except Exception as e:
            print(f"[错误] 电荷N-1失败: {e}")
            E_minus = -999.0
        init_state_half = ['up' if i % 2 == 0 else 'down' for i in range(L)]
        psi_half = MPS.from_product_state(sites_charge, init_state_half)
        try:
            E_half = dmrg.run(psi_half, model_charge, dmrg_params)['E']
        except Exception as e:
            print(f"[错误] 电荷N失败: {e}")
            E_half = -999.0
        if E_minus > -900 and E_half > -900:
            charge_gap = E_half - E_minus
        else:
            charge_gap = -999.0
    else:
        delta_s = -999.0
        A = -999.0
        C = -999.0
        charge_gap = -999.0
else:
    delta_s = -999.0
    A = -999.0
    C = -999.0
    charge_gap = -999.0

t_elapsed = time.time() - t0

print(f"\n周期链 L={L}:")
print(f"  自旋能隙 Δ_s = {delta_s:.6f}")
print(f"  前置因子 A = {A:.4f}")
print(f"  拓扑指数 C = {C:.6f}")
print(f"  电荷能隙 ΔE = {charge_gap:.4f}")
print(f"  耗时={t_elapsed:.0f}s")

with open(OUTPUT_FILE, "w") as f:
    f.write("L,boundary,spin_gap,A,C,charge_gap,timestamp\n")
    if A > -900:
        f.write(f"{L},periodic,{delta_s:.10f},{A:.6f},{C:.6f},{charge_gap:.6f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    else:
        f.write(f"{L},periodic,-999,-999,-999,-999,{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n" + "=" * 60)
print("[菜单二] 完成")
print("=" * 60)
print("\n对比基准（均匀开链，L=40）:")
print("  A = 3.0667, C = 0.958885, ΔE = 1.3140")
print("\n判决指南:")
print("  若周期链A与开链A显著不同(>5%) → 支持'关系先于实体'")
print("  若周期链A与开链A几乎相同(<5%) → 支持'实体优先'")
print("  无论结果如何，数据本身是诚实的。")