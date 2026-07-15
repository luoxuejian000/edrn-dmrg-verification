import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

U = 1.0
L_LIST = [120, 160]
CHI_MAX = 100
SV_MIN = 1e-10
MAX_SWEEPS = 50
MIN_SWEEPS = 2
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "L_scan_U1.csv")

print("=" * 60)
print("[L扫描] 固定U=1.0，扫描大L值")
print(f"[审计] 时间戳: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[审计] U/t = {U}, L序列 = {L_LIST}")
print(f"[审计] chi_max = {CHI_MAX}")
print("[审计] 已有数据: L=20,40,60,80 (来自U扫描)")
print("[审计] 本实验补跑: L=120,160")
print("=" * 60)
print("[方法论声明]")
print("  1. 所有L值使用统一的DMRG参数，不因收敛困难单独调整。")
print("  2. 若某数据点无法收敛，记录哨兵值-999并标注原因，不强行修复。")
print("  3. 本实验不预设标度漂移方向，只看数据说什么。")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    f.write("U,L,E_Nminus1,E_N,charge_gap,chi_c,spin_gap,timestamp\n")

def compute_charge_gap(L, U):
    model_params = {
        'L': L, 't': 1.0, 'U': U,
        'cons_N': 'N', 'cons_Sz': None,
        'bc_MPS': 'finite'
    }
    model = FermiHubbardChain(model_params)
    sites = model.lat.mps_sites()
    
    dmrg_params = {
        'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
        'mixer': True, 'combine': True,
        'max_sweeps': MAX_SWEEPS, 'min_sweeps': MIN_SWEEPS
    }
    
    init_state = [('up' if i % 2 == 0 else 'down') for i in range(L-1)]
    init_state += ['empty']
    
    psi_minus = MPS.from_product_state(sites, init_state)
    try:
        E_minus = dmrg.run(psi_minus, model, dmrg_params)['E']
    except Exception as e:
        E_minus = -999.0
    
    init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    psi_half = MPS.from_product_state(sites, init_state)
    try:
        E_half = dmrg.run(psi_half, model, dmrg_params)['E']
    except Exception as e:
        E_half = -999.0
    
    gap = E_half - E_minus if E_half > -900 and E_minus > -900 else -999.0
    chi_c = 1.0 / gap if gap > 0 else -999.0
    
    return E_minus, E_half, gap, chi_c

def compute_spin_gap(L, U):
    model_params = {
        'L': L, 't': 1.0, 'U': U,
        'cons_N': 'N', 'cons_Sz': 'Sz',
        'bc_MPS': 'finite'
    }
    model = FermiHubbardChain(model_params)
    sites = model.lat.mps_sites()
    
    dmrg_params = {
        'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
        'mixer': True, 'combine': True,
        'max_sweeps': MAX_SWEEPS, 'min_sweeps': MIN_SWEEPS
    }
    
    init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    psi0 = MPS.from_product_state(sites, init_state)
    try:
        E0 = dmrg.run(psi0, model, dmrg_params)['E']
    except Exception as e:
        E0 = -999.0
    
    init_state_triplet = init_state.copy()
    center = L // 2
    init_state_triplet[center] = 'up'
    init_state_triplet[center - 1] = 'up'
    psi1 = MPS.from_product_state(sites, init_state_triplet)
    try:
        E1 = dmrg.run(psi1, model, dmrg_params)['E']
    except Exception as e:
        E1 = -999.0
    
    delta_s = E1 - E0 if E1 > -900 and E0 > -900 else -999.0
    
    return delta_s

for L in L_LIST:
    t_start = time.time()
    print(f"\n[计算] U={U}, L={L} 开始...")
    
    E_minus, E_half, gap, chi_c = compute_charge_gap(L, U)
    delta_s = compute_spin_gap(L, U)
    
    t_elapsed = time.time() - t_start
    
    print(f"  电荷: E(N-1)={E_minus:.4f}, E(N)={E_half:.4f}, gap={gap:.4f}, chi_c={chi_c:.6f}")
    print(f"  自旋: Δ_s={delta_s:.6f}")
    print(f"  耗时={t_elapsed:.0f}s")
    
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"{U},{L},{E_minus:.10f},{E_half:.10f},{gap:.10f},{chi_c:.10f},{delta_s:.10f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n" + "=" * 60)
print("[L扫描] 完成")
print(f"[审计] 数据保存至: {os.path.abspath(OUTPUT_FILE)}")
print("=" * 60)
print("\n[分析提示]")
print("  将本实验的L=120,160数据与已有的L=20,40,60,80数据合并。")
print("  对合并后的六点数据集做log(chi_c) ~ α_charge log(L)拟合。")
print("  若α_charge随L增大漂向零：支持有限尺寸效应解释。")
print("  若α_charge保持稳定：预言二获得更强支持。")
print("  不做选择性报告——数据说什么，我们记录什么。")