import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

L_VALUES = [40, 60]
U = 4.0
DEFECT_LIST = [0.5, 0.8, 1.0, 1.2, 1.5]
CHI_MAX = 100
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "game1_charge_gap.csv")

print("=" * 60)
print("[游戏一·电荷能隙补跑] 单键缺陷电荷能隙计算")
print(f"U={U}, L序列: {L_VALUES}, 缺陷序列: {DEFECT_LIST}")
print(f"DMRG: chi_max={CHI_MAX}")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    f.write("L,defect_strength,E_Nminus1,E_N,charge_gap,chi_c,timestamp\n")

for L in L_VALUES:
    for ds in DEFECT_LIST:
        t_start = time.time()
        print(f"\n[运行] L={L}, 缺陷强度={ds}t ...")

        t_array = np.ones(L - 1)
        center_bond = L // 2 - 1
        t_array[center_bond] = ds

        model_params = {
            'L': L, 't': t_array, 'U': U,
            'cons_N': 'N', 'cons_Sz': None,
            'bc_MPS': 'finite',
        }
        model = FermiHubbardChain(model_params)
        sites = model.lat.mps_sites()

        dmrg_params = {
            'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
            'mixer': True, 'combine': True,
            'max_sweeps': 50, 'min_sweeps': 2,
        }

        init_state_minus = [('up' if i % 2 == 0 else 'down') for i in range(L-1)] + ['empty']
        psi_minus = MPS.from_product_state(sites, init_state_minus)
        try:
            E_minus = dmrg.run(psi_minus, model, dmrg_params)['E']
        except Exception as e:
            E_minus = -999.0
            print(f"  [警告] N=L-1 计算失败: {e}")

        init_state_half = ['up' if i % 2 == 0 else 'down' for i in range(L)]
        psi_half = MPS.from_product_state(sites, init_state_half)
        try:
            E_half = dmrg.run(psi_half, model, dmrg_params)['E']
        except Exception as e:
            E_half = -999.0
            print(f"  [警告] N=L 计算失败: {e}")

        gap = E_half - E_minus if (E_half > -900 and E_minus > -900) else -999.0
        chi_c = 1.0 / gap if gap > 0 else -999.0

        t_elapsed = time.time() - t_start
        print(f"  E(N-1)={E_minus:.6f}, E(N)={E_half:.6f}, gap={gap:.4f}, χ_c={chi_c:.6f}, 耗时={t_elapsed:.0f}s")

        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{L},{ds},{E_minus:.10f},{E_half:.10f},{gap:.10f},{chi_c:.10f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n" + "=" * 60)
print("[游戏一·电荷能隙补跑] 全部完成")
print(f"数据已保存至: {os.path.abspath(OUTPUT_FILE)}")
print("=" * 60)