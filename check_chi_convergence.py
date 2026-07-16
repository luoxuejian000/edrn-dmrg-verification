import numpy as np
import time
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

U = 1.0
L = 160
CHI_LIST = [100, 200, 300]
SV_MIN = 1e-10

print("=" * 60)
print(f"[χ外推验证] U={U}, L={L}")
print(f"[审计] χ序列: {CHI_LIST}")
print("=" * 60)

for chi in CHI_LIST:
    t0 = time.time()
    print(f"\n[计算] χ={chi} 开始...")
    
    model_params = {
        'L': L, 't': 1.0, 'U': U,
        'cons_N': 'N', 'cons_Sz': None,
        'bc_MPS': 'finite'
    }
    model = FermiHubbardChain(model_params)
    sites = model.lat.mps_sites()
    
    dmrg_p = {
        'trunc_params': {'chi_max': chi, 'svd_min': SV_MIN},
        'mixer': True, 'combine': True,
        'max_sweeps': 50, 'min_sweeps': 2
    }
    
    # 电荷能隙: N=L-1
    s = [('up' if i % 2 == 0 else 'down') for i in range(L-1)] + ['empty']
    Em = dmrg.run(MPS.from_product_state(sites, s), model, dmrg_p)['E']
    
    # 电荷能隙: N=L
    s = [('up' if i % 2 == 0 else 'down') for i in range(L)]
    Eh = dmrg.run(MPS.from_product_state(sites, s), model, dmrg_p)['E']
    
    gap = Eh - Em
    chi_c = 1.0 / gap if gap > 0 else -999
    
    # 自旋能隙
    model2 = FermiHubbardChain({
        'L': L, 't': 1.0, 'U': U,
        'cons_N': 'N', 'cons_Sz': 'Sz',
        'bc_MPS': 'finite'
    })
    sites2 = model2.lat.mps_sites()
    
    s0 = [('up' if i % 2 == 0 else 'down') for i in range(L)]
    E0 = dmrg.run(MPS.from_product_state(sites2, s0), model2, dmrg_p)['E']
    
    s1 = s0.copy()
    c = L // 2
    s1[c] = 'up'
    s1[c-1] = 'up'
    E1 = dmrg.run(MPS.from_product_state(sites2, s1), model2, dmrg_p)['E']
    
    ds = E1 - E0
    
    t = time.time() - t0
    print(f"  χ={chi}: gap={gap:.6f}, χ_c={chi_c:.6f}, Δ_s={ds:.6f}, 耗时={t:.0f}s")

print("\n" + "=" * 60)
print("[χ外推验证] 完成")
print("若χ_c和Δ_s在χ=200和300时基本不变，则χ=100已收敛。")
print("=" * 60)