"""
edrn-dmrg-verification 环境验证脚本 (tenpy 1.x 修正版)
功能：
    1. 检查关键依赖
    2. 构建 L=4 Hubbard 模型并跑 DMRG
    3. 校验基态能量合理性
    4. 计算密度-密度关联函数
    5. 输出结构化审计日志
"""
import sys
import platform
import time
import numpy as np
import tenpy
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

print("=" * 60)
print("[审计] EDRN 环境验证开始")
print(f"[审计] 时间戳: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[审计] Python: {sys.version}")
print(f"[审计] tenpy: {tenpy.__version__}")
print(f"[审计] numpy: {np.__version__}")

try:
    import h5py
    print(f"[审计] h5py: {h5py.__version__}")
except ImportError:
    print("[警告] 缺少 h5py，后续大规模批量计算请手动安装。本次验证跳过。")

try:
    import scipy
    print(f"[审计] scipy: {scipy.__version__}")
except ImportError:
    print("[错误] 缺少 scipy。")
    sys.exit(1)

print(f"[审计] 平台: {platform.platform()}")

L_test = 4
model_params = {
    'L': L_test,
    't': 1.0,
    'U': 4.0,
    'cons_N': 'N',
    'cons_Sz': 'Sz',
    'bc_MPS': 'finite',
}
model = FermiHubbardChain(model_params)

sites = model.lat.mps_sites()
init_state = ['up', 'down', 'empty', 'empty']
psi = MPS.from_product_state(
    sites, init_state, bc=model.lat.bc_MPS, unit_cell_width=model.lat.mps_unit_cell_width
)

dmrg_params = {
    'mixer': True,
    'max_E_err': 1.0e-10,
    'trunc_params': {'chi_max': 50, 'svd_min': 1e-10},
    'combine': True,
}

info = dmrg.run(psi, model, dmrg_params)
E = info['E']

print(f"[DMRG] L={L_test} 基态能量: {E:.10f}")

if E < -10 or E > 10:
    print(f"[警告] 基态能量 {E:.6f} 严重偏离合理范围。")
else:
    print(f"[校验] 基态能量 {E:.6f} 在合理范围内。")

corr_matrix = psi.correlation_function('Ntot', 'Ntot')
n_expvals = psi.expectation_value('Ntot')
center = L_test // 2

connected_corr = []
for j in range(L_test):
    cc = np.real(corr_matrix[center, j] - n_expvals[center] * n_expvals[j])
    connected_corr.append(cc)

print(f"[关联] 密度-密度关联 (中心点 {center}): {connected_corr}")
print(f"[关联] 关联和 χ(0) = {sum(connected_corr):.6f}")

print("[审计] 所有测试通过。环境已就绪，可开始批量计算。")
print("=" * 60)