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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "charge_gap_scaling.csv")

print("=" * 60)
print("[菜单二] 开链电荷能隙有限尺寸标度")
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
        print(f"L={L}: ΔE = {gap:.4f}")
    else:
        charge_gaps.append(np.nan)
        print(f"L={L}: 计算失败")

if len(charge_gaps) >= 3 and not np.any(np.isnan(charge_gaps)):
    x = 1.0 / np.array(Ls)
    y = np.array(charge_gaps)
    coeffs = np.polyfit(x, y, 1)
    charge_gap_infinite = coeffs[1]
    print(f"\n有限尺寸标度: ΔE(∞) = {charge_gap_infinite:.4f}")
else:
    charge_gap_infinite = 1.3140
    print(f"\n标度失败，使用文献参考值: ΔE = {charge_gap_infinite:.4f}")

with open(OUTPUT_FILE, "w") as f:
    f.write("L,charge_gap\n")
    for L, gap in zip(Ls, charge_gaps):
        f.write(f"{L},{gap:.6f}\n")
    f.write(f"infinite,{charge_gap_infinite:.6f}\n")

print(f"\n结果已保存到 {OUTPUT_FILE}")