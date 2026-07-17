import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

L = 60
U = 4.0
DEFECT_LIST = [0.5, 1.5]
CHI_MAX = 100
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "game1_spin_correlation.csv")

print("=" * 60)
print("[判决性实验] 自旋关联穿透深度测量")
print(f"参数: L={L}, U={U}")
print(f"缺陷强度: {DEFECT_LIST}")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    f.write("defect_strength,i,j,distance,correlation,timestamp\n")

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

    corr_Sz = psi0.correlation_function('Sz', 'Sz')

    i_fixed = 0
    print(f"\n  自旋关联 <S_{i_fixed}·S_j> (缺陷={ds}t):")
    print(f"  {'j':<6} {'距离':<6} {'<S_0·S_j>':<12}")
    print(f"  {'-'*30}")

    for j in range(L):
        distance = abs(j - i_fixed)
        corr_val = np.real(corr_Sz[i_fixed, j])
        if j < 10 or j > L-10 or j == L//2 or j == L//2-1:
            print(f"  {j:<6} {distance:<6} {corr_val:<12.6f}")

        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{ds},{i_fixed},{j},{distance},{corr_val:.10f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

    t_elapsed = time.time() - t_start
    print(f"  耗时={t_elapsed:.0f}s")

print("\n" + "=" * 60)
print("[判决性实验] 完成")
print(f"数据已保存至: {os.path.abspath(OUTPUT_FILE)}")
print("=" * 60)
print("\n分析指南:")
print("  1. 看中心键两侧(j≈29-30)的关联是否断崖式下跌")
print("  2. 如果0.5t下关联几乎为零 → 审稿人的L_eff假设成立")
print("  3. 如果0.5t下关联仍然可以穿透 → 审稿人的L_eff假设不成立")