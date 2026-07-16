"""
实验：自旋结构因子 S(q) 的标度检验
目标：验证自旋扇区是否存在与 η=1/4 一致的临界标度
理论预言：S(q) ∼ q^{-1/2}（在对数坐标下斜率为 -0.5）

严格遵循：
- 只看病，不开方：只诊断数据，不预设结论
- 严防工具理性悖论：不选择性拟合，不强行制造证据
- 诚实记录所有数据
"""
import numpy as np
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg
import time
import os

L_LIST = [20, 40, 60, 80, 100]
U = 4.0
CHI_MAX = 100
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "spin_structure_factor.csv")

print("=" * 60)
print("[审计] 自旋结构因子标度检验")
print(f"[审计] 时间戳: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[审计] L 序列: {L_LIST}")
print(f"[审计] U/t = {U}, chi_max = {CHI_MAX}")
print(f"[审计] 理论预言：S(q) ∼ q^{-1/2}，即 log S(q) vs log q 斜率 = -0.5")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    f.write("L,q_index,q_value,S_q,log_q,log_S_q,timestamp\n")

for L in L_LIST:
    t_start = time.time()
    print(f"\n[计算] L={L} 开始...")
    
    model_params = {
        'L': L, 't': 1.0, 'U': U,
        'cons_N': 'N', 'cons_Sz': 'Sz',
        'bc_MPS': 'finite'
    }
    model = FermiHubbardChain(model_params)
    sites = model.lat.mps_sites()
    
    init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
    psi = MPS.from_product_state(sites, init_state)
    
    dmrg_params = {
        'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
        'mixer': True,
        'combine': True,
        'max_sweeps': 50,
        'min_sweeps': 2
    }
    
    result = dmrg.run(psi, model, dmrg_params)
    E0 = result['E']
    print(f"  基态能量: {E0:.6f}")
    
    corr_matrix = psi.correlation_function('Sz', 'Sz')
    
    q_values = []
    S_q_values = []
    
    for n in range(1, L // 2 + 1):
        q = 2 * np.pi * n / L
        
        S_q = 0.0
        for i in range(L):
            for j in range(L):
                phase = np.cos(q * (i - j))
                S_q += phase * corr_matrix[i, j]
        S_q /= L
        
        q_values.append(q)
        S_q_values.append(S_q)
    
    for idx, (q_val, S_q_val) in enumerate(zip(q_values, S_q_values)):
        log_q = np.log(q_val)
        log_S_q = np.log(S_q_val) if S_q_val > 0 else np.nan
        
        if not np.isnan(log_S_q):
            print(f"  q={q_val:.4f}, S(q)={S_q_val:.6f}, log q={log_q:.4f}, log S(q)={log_S_q:.4f}")
        
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{L},{idx},{q_val:.10f},{S_q_val:.10f},{log_q:.10f},{log_S_q},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    
    t_elapsed = time.time() - t_start
    print(f"[完成] L={L}, 耗时={t_elapsed:.1f}s")

print("\n" + "=" * 60)
print("[审计] 实验完成")
print(f"[审计] 数据保存至: {os.path.abspath(OUTPUT_FILE)}")
print()
print("[分析提示]")
print("1. 对每个L，提取小q区域的 log S(q) vs log q 斜率")
print("2. 若斜率 ≈ -0.5，则 η=1/4 在自旋扇区获得动量空间验证")
print("3. 若斜率偏离 -0.5，诚实记录偏差，分析可能原因")
print("4. 不选择性报告数据——所有q值的结果全部呈现")
print("=" * 60)