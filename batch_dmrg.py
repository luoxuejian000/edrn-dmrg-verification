import sys
import os
import time
import numpy as np
import tenpy
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

L_LIST = [20, 40, 60, 80, 100, 120, 160, 200]
U_OVER_T = 4.0
CHI_MAX = 200
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chi0_vs_L.csv")

print("=" * 60)
print("[审计] EDRN 批量 DMRG 计算开始 (tenpy 1.1.0)")
print(f"[审计] 时间戳: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[审计] tenpy: {tenpy.__version__}")
print(f"[审计] L 序列: {L_LIST}")
print(f"[审计] U/t = {U_OVER_T}, chi_max = {CHI_MAX}")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    f.write("L,chi_max,U_over_t,energy,avg_density,timestamp\n")

for L in L_LIST:
    t_start = time.time()
    print(f"\n[计算] L={L} 开始...")

    model_params = {
        'L': L,
        't': 1.0,
        'U': U_OVER_T,
        'cons_N': 'N',
        'cons_Sz': 'Sz',
        'bc_MPS': 'finite',
    }
    model = FermiHubbardChain(model_params)

    sites = model.lat.mps_sites()
    N_particles = L // 2
    N_up = N_particles // 2
    N_down = N_particles // 2
    init_state = ['up'] * N_up + ['down'] * N_down + ['empty'] * (L - N_particles)
    psi = MPS.from_product_state(
        sites, init_state, bc=model.lat.bc_MPS, unit_cell_width=model.lat.mps_unit_cell_width
    )

    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.0e-10,
        'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
        'combine': True,
    }

    info = dmrg.run(psi, model, dmrg_params)
    E = info['E']

    n_expvals = psi.expectation_value('Ntot')
    avg_density = np.mean(n_expvals)

    t_elapsed = time.time() - t_start
    print(f"[完成] L={L}: E={E:.6f}, 平均密度={avg_density:.6f}, 耗时={t_elapsed:.1f}s")

    with open(OUTPUT_FILE, "a") as f:
        f.write(f"{L},{CHI_MAX},{U_OVER_T},{E:.10f},{avg_density:.10f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

    del model, psi

print("\n" + "=" * 60)
print("[审计] 批量计算完成")
print(f"[审计] 数据已保存至: {os.path.abspath(OUTPUT_FILE)}")
print("=" * 60)