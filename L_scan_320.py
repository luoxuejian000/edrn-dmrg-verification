import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

U = 1.0
L = 320
CHI_MAX = 100
SV_MIN = 1e-10
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "L_scan_U1_L320.csv")

print("=" * 60)
print(f"[L扫描] 固定U={U}, L={L}")
print(f"[审计] 时间戳: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[审计] chi_max = {CHI_MAX}")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    f.write("U,L,E_Nminus1,E_N,charge_gap,chi_c,spin_gap,timestamp\n")

t0 = time.time()
print(f"\n[计算] U={U}, L={L} 开始...")

model = FermiHubbardChain({
    'L': L, 't': 1.0, 'U': U,
    'cons_N': 'N', 'cons_Sz': None,
    'bc_MPS': 'finite'
})
sites = model.lat.mps_sites()

dmrg_p = {
    'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
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
print(f"  电荷: gap={gap:.4f}, chi_c={chi_c:.6f}")
print(f"  自旋: Δ_s={ds:.6f}")
print(f"  耗时={t:.0f}s")

with open(OUTPUT_FILE, "a") as f:
    f.write(f"{U},{L},{Em:.10f},{Eh:.10f},{gap:.10f},{chi_c:.10f},{ds:.10f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

print("\n" + "=" * 60)
print(f"[L扫描] 完成。数据: {os.path.abspath(OUTPUT_FILE)}")
print("=" * 60)