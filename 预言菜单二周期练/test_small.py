import numpy as np
import time
import os
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.networks.mps import MPS
import tenpy.algorithms.dmrg as dmrg

L = 8
U = 4.0
CHI_MAX = 100
SV_MIN = 1e-10

print(f"测试小系统 L={L}")

model_params = {
    'lattice': 'Chain',
    'L': L,
    't': 1.0,
    'U': U,
    'cons_N': 'N',
    'cons_Sz': None,
    'bc_MPS': 'finite',
    'bc_x': 'periodic',
}
model = FermiHubbardModel(model_params)
sites = model.lat.mps_sites()

dmrg_params = {
    'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SV_MIN},
    'mixer': True,
    'combine': True,
    'max_sweeps': 50,
    'min_sweeps': 2,
    'verbose': 2,
}

init_state = ['up' if i % 2 == 0 else 'down' for i in range(L)]
psi = MPS.from_product_state(sites, init_state)
print(f"\n[运行] N=L 扇区...")
t0 = time.time()
try:
    result = dmrg.run(psi, model, dmrg_params)
    E = result['E']
    print(f"  E(N) = {E:.6f}")
except Exception as e:
    E = -999.0
    print(f"  [警告] 计算失败: {e}")

t_elapsed = time.time() - t0
print(f"  耗时={t_elapsed:.0f}s")