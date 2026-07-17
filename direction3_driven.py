import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

L = 40
U = 4.0
CHI_MAX = 100
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "direction3_driven.csv")

print("=" * 60)
print("[方向三] 矛盾动力计算——非平衡跨扇区矛盾守恒")
print(f"参数: L={L}, U={U}")
print(f"驱动: 中心键在 0.5t 和 1.5t 之间振荡")
print(f"DMRG: chi_max={CHI_MAX}")
print("=" * 60)
print("[方法论声明]")
print("  本实验检验非平衡条件下跨扇区守恒是否依然成立。")
print("  不预设结论，不对齐理论，只让数据说话。")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

defect_values = [0.5, 0.75, 1.0, 1.25, 1.5]

with open(OUTPUT_FILE, "w") as f:
    f.write("t_defect,spin_gap,A,charge_gap,chi_c,timestamp\n")

for ds in defect_values:
    t_start = time.time()
    print(f"\n[快照] t_defect = {ds}t ...")

    t_array = np.ones(L - 1)
    center_bond = L // 2 - 1
    t_array[center_bond] = ds

    model_spin = FermiHubbardChain({
        'L': L, 't': t_array, 'U': U,
        'cons_N': 'N', 'cons_Sz': 'Sz',
        'bc_MPS': 'finite',
    })
    sites_spin = model_spin.lat.mps_sites()

    dmrg_params = {
        'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
        'mixer': True, 'combine': True,
        'max_sweeps': 50, 'min_sweeps': 2,
    }

    init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    psi0 = MPS.from_product_state(sites_spin, init_state)
    E0 = dmrg.run(psi0, model_spin, dmrg_params)['E']

    init_state_triplet = init_state.copy()
    center = L // 2
    init_state_triplet[center] = 'up'
    init_state_triplet[center - 1] = 'up'
    psi1 = MPS.from_product_state(sites_spin, init_state_triplet)
    E1 = dmrg.run(psi1, model_spin, dmrg_params)['E']

    spin_gap = E1 - E0
    A = spin_gap * L

    model_charge = FermiHubbardChain({
        'L': L, 't': t_array, 'U': U,
        'cons_N': 'N', 'cons_Sz': None,
        'bc_MPS': 'finite',
    })
    sites_charge = model_charge.lat.mps_sites()

    init_minus = [('up' if i % 2 == 0 else 'down') for i in range(L-1)] + ['empty']
    psi_minus = MPS.from_product_state(sites_charge, init_minus)
    E_minus = dmrg.run(psi_minus, model_charge, dmrg_params)['E']

    init_half = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    psi_half = MPS.from_product_state(sites_charge, init_half)
    E_half = dmrg.run(psi_half, model_charge, dmrg_params)['E']

    charge_gap = E_half - E_minus
    chi_c = 1.0 / charge_gap if charge_gap > 0 else -999.0

    t_elapsed = time.time() - t_start
    print(f"  Δ_s={spin_gap:.6f}, A={A:.4f}, ΔE={charge_gap:.4f}, χ_c={chi_c:.4f}, 耗时={t_elapsed:.0f}s")

    with open(OUTPUT_FILE, "a") as f:
        f.write(f"{ds},{spin_gap:.10f},{A:.6f},{charge_gap:.10f},{chi_c:.6f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n" + "=" * 60)
print("[方向三] 快照扫描完成")
print(f"数据已保存至: {os.path.abspath(OUTPUT_FILE)}")
print("=" * 60)
print("\n分析指南:")
print("  1. 计算每个快照的 α_charge + α_spin")
print("  2. 若和在所有快照中保持 ≈ -1 → 非平衡跨扇区守恒成立")
print("  3. 若和随驱动变化 → 记录偏离规律，诚实报告")