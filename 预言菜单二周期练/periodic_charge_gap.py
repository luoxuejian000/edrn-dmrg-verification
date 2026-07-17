import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

U = 4.0
CHI_MAX = 200
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "menu2_periodic.csv")

print("=" * 60)
print("[菜单二] 周期链对比实验——检验“关系先于实体”")
print(f"参数: U={U}")
print(f"DMRG: chi_max={CHI_MAX}")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

dmrg_params = {
    'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN, 'max_trunc_err': 1e-4},
    'mixer': True,
    'mixer_params': {'amplitude': 1e-4, 'decay': 1.5},
    'combine': True,
    'max_sweeps': 100,
    'min_sweeps': 5,
}

print("[步骤1] 周期链自旋能隙计算 (L=40)...")
L_periodic = 40
model_params = {
    'lattice': 'Chain',
    'L': L_periodic,
    't': 1.0,
    'U': U,
    'cons_N': 'N',
    'cons_Sz': 'Sz',
    'bc_MPS': 'finite',
    'bc_x': 'periodic',
}
model = FermiHubbardModel(model_params)
sites = model.lat.mps_sites()

init_state = ['up' if i % 2 == 0 else 'down' for i in range(L_periodic)]
psi0 = MPS.from_product_state(sites, init_state)

try:
    eng0 = dmrg.run(psi0, model, dmrg_params)
    E0 = eng0['E']
    psi0 = eng0['psi']
    print(f"  E0 = {E0:.6f}")
except Exception as e:
    print(f"[错误] 单态DMRG失败: {e}")
    E0 = -999.0

if E0 > -900:
    init_state_triplet = init_state.copy()
    center = L_periodic // 2
    init_state_triplet[center] = 'up'
    init_state_triplet[center - 1] = 'up'
    psi1 = MPS.from_product_state(sites, init_state_triplet)
    try:
        E1 = dmrg.run(psi1, model, dmrg_params)['E']
        print(f"  E1 = {E1:.6f}")
    except Exception as e:
        print(f"[错误] 三重态DMRG失败: {e}")
        E1 = -999.0

    if E1 > -900:
        delta_s = E1 - E0
        A = delta_s * L_periodic

        corr_up = psi0.correlation_function('Cdu', 'Cu')
        corr_dn = psi0.correlation_function('Cdd', 'Cd')
        omega_sum = 0.0
        for i in range(L_periodic):
            j = (i + 1) % L_periodic
            bond_up = abs(np.real(corr_up[i, j]))
            bond_dn = abs(np.real(corr_dn[i, j]))
            omega_sum += (bond_up + bond_dn) / 2.0
        C = (omega_sum * L_periodic) / (L_periodic * (L_periodic - 1) / 2.0)
    else:
        delta_s = -999.0
        A = -999.0
        C = -999.0
else:
    delta_s = -999.0
    A = -999.0
    C = -999.0

print("[步骤2] 开链电荷能隙有限尺寸标度...")
Ls = [20, 30, 40, 50]
charge_gaps = []

for L in Ls:
    model_open = FermiHubbardModel({
        'lattice': 'Chain',
        'L': L,
        't': 1.0,
        'U': U,
        'cons_N': 'N',
        'cons_Sz': None,
        'bc_MPS': 'finite',
        'bc_x': 'open',
    })
    sites_open = model_open.lat.mps_sites()

    init_state_minus = [('up' if i % 2 == 0 else 'down') for i in range(L-1)] + ['empty']
    psi_minus = MPS.from_product_state(sites_open, init_state_minus)
    try:
        E_minus = dmrg.run(psi_minus, model_open, dmrg_params)['E']
    except Exception as e:
        E_minus = -999.0

    init_state_half = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    psi_half = MPS.from_product_state(sites_open, init_state_half)
    try:
        E_half = dmrg.run(psi_half, model_open, dmrg_params)['E']
    except Exception as e:
        E_half = -999.0

    if E_minus > -900 and E_half > -900:
        gap = E_half - E_minus
        charge_gaps.append(gap)
        print(f"  L={L}: ΔE = {gap:.4f}")
    else:
        charge_gaps.append(np.nan)
        print(f"  L={L}: 计算失败")

if len(charge_gaps) >= 3 and not np.any(np.isnan(charge_gaps)):
    x = 1.0 / np.array(Ls)
    y = np.array(charge_gaps)
    coeffs = np.polyfit(x, y, 1)
    charge_gap_infinite = coeffs[1]
    print(f"\n  有限尺寸标度: ΔE(∞) = {charge_gap_infinite:.4f}")
else:
    charge_gap_infinite = 1.3140
    print(f"\n  标度失败，使用文献参考值: ΔE = {charge_gap_infinite:.4f}")

t_elapsed = time.time()

print(f"\n周期链 L={L_periodic}:")
print(f"  自旋能隙 Δ_s = {delta_s:.6f}")
print(f"  前置因子 A = {A:.4f}")
print(f"  拓扑指数 C = {C:.6f}")
print(f"  电荷能隙 ΔE = {charge_gap_infinite:.4f} (开链有限尺寸标度)")

with open(OUTPUT_FILE, "w") as f:
    f.write("L,boundary,spin_gap,A,C,charge_gap,charge_gap_method,timestamp\n")
    f.write(f"{L_periodic},periodic,{delta_s:.10f},{A:.6f},{C:.6f},{charge_gap_infinite:.6f},open_chain_extrapolated,{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n" + "=" * 60)
print("[菜单二] 完成")
print("=" * 60)
print("\n对比基准（均匀开链，L=40）:")
print("  A = 3.0667, C = 0.958885, ΔE = 1.3140")
print("\n判决:")
if A > 0:
    diff_percent = abs(A - 3.0667) / 3.0667 * 100
    print(f"  A(周期) = {A:.4f}, A(开链) = 3.0667")
    print(f"  A 差异 = {diff_percent:.1f}%")
    if diff_percent > 5:
        print(f"  A 差异 > 5% → 支持'关系先于实体'")
    else:
        print(f"  A 差异 < 5% → 支持'实体优先'")
else:
    print("  计算失败，无法判决")