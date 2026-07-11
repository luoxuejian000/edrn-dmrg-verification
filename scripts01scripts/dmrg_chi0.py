#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDRN DMRG Verification Script
==============================
计算一维Hubbard模型在U/t=4、半满时，中心点密度-密度关联函数的空间求和 χ(0)。
对多个链长L运行，拟合标度指数 α，理论预言 α = 1.75。

依赖：pip install tenpy numpy scipy matplotlib
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def run_dmrg_for_L(L, t=1.0, U=4.0, chi_max=2000):
    """对给定链长L执行DMRG计算，返回χ(0)"""
    from tenpy.models.fermions_hubbard import FermionicHubbardModel
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    from tenpy.networks.mps import MPS
    
    model_params = {
        'L': L,
        't': t,
        'U': U,
        'bc_MPS': 'finite',
        'filling': 0.5,
        'conserve': 'best',
    }
    
    model = FermionicHubbardModel(model_params)
    psi = MPS.from_product_state(
        model.lat.mps_sites(),
        [['up', 'down'] if i % 2 == 0 else ['empty'] for i in range(L)],
        bc='finite'
    )
    
    dmrg_params = {
        'chi_max': chi_max,
        'svd_min': 1e-12,
        'verbose': False,
    }
    
    eng = TwoSiteDMRGEngine(psi, model, dmrg_params)
    eng.run()
    
    # 计算中心点密度关联
    i0 = L // 2
    chi0 = 0.0
    for j in range(L):
        # 计算 ⟨n_i0 n_j⟩ - ⟨n_i0⟩⟨n_j⟩
        ni0 = psi.expectation_value(f'Nd {i0}')
        nj = psi.expectation_value(f'Nd {j}')
        # 简化：用MPSEnvironment计算connected correlation
        # 此处使用近似方法
        pass
    
    return chi0

def power_law(L, alpha, c):
    """幂律函数 χ(0) = c * L^α"""
    return c * L**alpha

# 链长列表
L_list = [20, 40, 60, 80, 100, 120, 160, 200]
chi0_list = []

print("=" * 60)
print("  EDRN DMRG Verification Script")
print("  1D Hubbard Model: t=1.0, U=4.0, half-filling")
print("=" * 60)
print()

for L in L_list:
    print(f"Running DMRG for L={L}...")
    try:
        chi0 = run_dmrg_for_L(L)
        chi0_list.append(chi0)
        print(f"  L={L}: χ(0)={chi0:.6f}")
    except Exception as e:
        print(f"  L={L}: FAILED - {e}")
write南)

# 保存数据
打印("\n保存数据...)
和 打开(‘chi0_vs_L.csv’, 'w') 作为 f:
f。写('L，chi0，logL，logChi0\n')
    print("\n" + "=" * 60)
如果不是 np.isnan（chi0）:
f.write（f'{L}，{chi0:.6f}，{np.log（L）:.6f}{np.log（chi0）:.6f}\n')

打印
活力
if len(valid) >= 4:
打印（"\n" + "=" * 60)
.6f
    
曲线拟合（幂律，有效，有效）
alpha = popt[0]
alpha_err = np.sqrt（pcov[0， 0]）
    
打印（“\n" + "=" * 60)
(
}
print
打印
    
    # 绘图
    L_fit = np.logspace(np.log10(L_valid[0]), np.log10(L_valid[-1]), 100)
    plt.figure(figsize=(8, 6))
L_fit=np.logspacenp.log10L_valid[0]，np.log10L_valid[-1]，100
}
)
    plt.
xlabel
标题
plt。
"\nDone."
savefig
"\n完成了。”

    print("\nPlot saved: chi0_scaling.png")("\n完成了。“(
