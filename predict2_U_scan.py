import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

U_LIST = [0.5, 1.0, 2.0]
L_LIST = [20, 40, 60, 80]
CHI_MAX = 100
SV_MIN = 1e-10
MAX_SWEEPS = 50
MIN_SWEEPS = 2
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predict2_U_scan_v2.csv")

print("=" * 60)
print("[预言二] 有效临界U值扫描：电荷与自旋标度行为")
print(f"[审计] 时间戳: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[审计] U/t 序列: {U_LIST}")
print(f"[审计] L 序列: {L_LIST}")
print(f"[审计] chi_max = {CHI_MAX}, svd_min = {SV_MIN}")
print("=" * 60)
print("[方法论声明]")
print("  1. 所有U值和L值使用统一的DMRG参数，不因收敛困难单独调整。")
print("  2. 本实验不预设标度交换一定存在，只看数据说什么。")
print("  3. 若某数据点无法收敛，记录哨兵值-999并标注原因，不强行修复。")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    f.write("U,L,E_Nminus1,E_N,charge_gap,chi_c,spin_gap,alpha_c_estimate,alpha_s_estimate,status,timestamp\n")

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
        'mixer': True,
        'combine': True,
        'max_sweeps': MAX_SWEEPS,
        'min_sweeps': MIN_SWEEPS
    }
    
    if L-1 <= L:
        init_state = [('up' if i % 2 == 0 else 'down') for i in range(L-1)]
        init_state += ['empty'] * (L - (L-1))
    else:
        init_state = [('up' if i % 2 == 0 else 'down') for i in range(L)]
        extra = (L-1) - L
        init_state += ['full'] * extra
    
    psi_minus = MPS.from_product_state(sites, init_state)
    try:
        E_minus = dmrg.run(psi_minus, model, dmrg_params)['E']
        status_minus = "OK"
    except Exception as e:
        E_minus = -999.0
        status_minus = f"FAIL:{str(e)[:50]}"
    
    init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    psi_half = MPS.from_product_state(sites, init_state)
    try:
        E_half = dmrg.run(psi_half, model, dmrg_params)['E']
        status_half = "OK"
    except Exception as e:
        E_half = -999.0
        status_half = f"FAIL:{str(e)[:50]}"
    
    gap = E_half - E_minus if E_half > -900 and E_minus > -900 else -999.0
    chi_c = 1.0 / gap if gap > 0 else -999.0
    status = "OK" if status_minus == "OK" and status_half == "OK" else f"MINUS:{status_minus}|HALF:{status_half}"
    
    return E_minus, E_half, gap, chi_c, status


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
        'mixer': True,
        'combine': True,
        'max_sweeps': MAX_SWEEPS,
        'min_sweeps': MIN_SWEEPS
    }
    
    init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    psi0 = MPS.from_product_state(sites, init_state)
    try:
        E0 = dmrg.run(psi0, model, dmrg_params)['E']
        status0 = "OK"
    except Exception as e:
        E0 = -999.0
        status0 = f"FAIL:{str(e)[:50]}"
    
    init_state_triplet = init_state.copy()
    center = L // 2
    init_state_triplet[center] = 'up'
    init_state_triplet[center - 1] = 'up'
    psi1 = MPS.from_product_state(sites, init_state_triplet)
    try:
        E1 = dmrg.run(psi1, model, dmrg_params)['E']
        status1 = "OK"
    except Exception as e:
        E1 = -999.0
        status1 = f"FAIL:{str(e)[:50]}"
    
    delta_s = E1 - E0 if E1 > -900 and E0 > -900 else -999.0
    status = "OK" if status0 == "OK" and status1 == "OK" else f"S0:{status0}|S1:{status1}"
    
    return delta_s, status


def estimate_scaling_exponent(values_dict, L_list):
    valid_pairs = [(L, val) for L, val in values_dict.items() if val > -900 and val > 0]
    if len(valid_pairs) < 2:
        return -999.0
    
    log_L = np.log([p[0] for p in valid_pairs])
    log_val = np.log([p[1] for p in valid_pairs])
    
    slope = np.polyfit(log_L, log_val, 1)[0]
    return slope


all_chi_c = {}
all_delta_s = {}

for U in U_LIST:
    all_chi_c[U] = {}
    all_delta_s[U] = {}
    
    for L in L_LIST:
        t_start = time.time()
        print(f"\n[计算] U/t={U}, L={L} 开始...")
        
        E_minus, E_half, gap, chi_c, status_c = compute_charge_gap(L, U)
        all_chi_c[U][L] = chi_c
        
        delta_s, status_s = compute_spin_gap(L, U)
        all_delta_s[U][L] = delta_s
        
        alpha_c_est = estimate_scaling_exponent(all_chi_c[U], L_LIST)
        alpha_s_est = estimate_scaling_exponent(all_delta_s[U], L_LIST)
        
        t_elapsed = time.time() - t_start
        
        data_status = "OK"
        if chi_c < -900 or delta_s < -900:
            data_status = "FLAGGED"
        if "FAIL" in status_c or "FAIL" in status_s:
            data_status = "CONVERGENCE_ISSUE"
        
        print(f"  电荷: E(N-1)={E_minus:.4f}, E(N)={E_half:.4f}, gap={gap:.4f}, chi_c={chi_c:.6f}")
        print(f"  自旋: Δ_s={delta_s:.6f}")
        print(f"  当前估算: α_c≈{alpha_c_est:.3f}, α_s≈{alpha_s_est:.3f}")
        print(f"  状态: {data_status}, 耗时={t_elapsed:.0f}s")
        
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{U},{L},{E_minus:.10f},{E_half:.10f},{gap:.10f},{chi_c:.10f},{delta_s:.10f},{alpha_c_est:.6f},{alpha_s_est:.6f},{data_status},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n" + "=" * 60)
print("[预言二] 扫描完成")
print(f"[审计] 数据已保存至: {os.path.abspath(OUTPUT_FILE)}")
print("=" * 60)

print("\n[汇总] 各U值下的标度指数（基于当前L序列的实时估算）:")
print(f"{'U/t':<8} {'α_charge':<12} {'α_spin':<12} {'物理状态':<20}")
print("-" * 52)
for U in U_LIST:
    alpha_c = estimate_scaling_exponent(all_chi_c[U], L_LIST)
    alpha_s = estimate_scaling_exponent(all_delta_s[U], L_LIST)
    
    if alpha_c > 0.1:
        state = "金属/临界"
    elif alpha_c < -0.1:
        state = "绝缘体"
    else:
        state = "过渡/趋平"
    
    print(f"{U:<8} {alpha_c:<12.4f} {alpha_s:<12.4f} {state:<20}")

print("\n[判决指南]")
print("  若α_charge从正值(U=0.5)单调下降到零或负值(U=4)，且α_spin始终保持≈-1：")
print("    → 支持预言二：存在有效临界U，电荷与自旋标度行为发生交换。")
print("  若α_charge在所有U值下都趋近零：")
print("    → 否定预言二：电荷从未显示金属标度，或Mott能隙在所有U>0下已打开。")
print("  若α_spin随U显著变化：")
print("    → 修正预言二：自旋临界性的普适范围需要重新界定。")
print("\n[方法论提醒]")
print("  以上只是基于4个L值的实时估算，不是最终标度指数。")
print("  每个U值的精确标度指数需要单独做log-log拟合。")
print("  不做选择性报告。不做数据对齐。只看病，不开方。")
print("=" * 60)