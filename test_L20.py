import numpy as np
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

L = 20

model = FermiHubbardChain({'L': L, 't': 1.0, 'U': 4.0, 'cons_N': 'N', 'cons_Sz': 'Sz', 'bc_MPS': 'finite'})
sites = model.lat.mps_sites()

init_state = ['up', 'down', 'up', 'down', 'up', 'down', 'up', 'down', 'up', 'down', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty']
psi = MPS.from_product_state(
    sites, init_state, bc=model.lat.bc_MPS, unit_cell_width=model.lat.mps_unit_cell_width
)

dmrg_params = {
    'mixer': True,
    'max_E_err': 1.0e-10,
    'trunc_params': {'chi_max': 200, 'svd_min': 1.0e-10},
    'combine': True,
}

info = dmrg.run(psi, model, dmrg_params)
E = info['E']

corr = psi.correlation_function('Ntot', 'Ntot')
nexp = psi.expectation_value('Ntot')

print(f'L={L}: E={E:.10f}')
print(f'总粒子数: {np.sum(nexp):.6f}')
print(f'平均密度: {np.mean(nexp):.6f}')

print('\n关联函数矩阵 (中心行):')
for j in range(L):
    print(f'  j={j:2d}: <n_10 n_{j}> = {corr[10,j]:.10f}, <n_10><n_{j}> = {nexp[10]*nexp[j]:.10f}, 差 = {corr[10,j]-nexp[10]*nexp[j]:.15f}')

chi0_direct = np.sum(np.real(corr[10, :]))
chi0_disconnected = np.sum(np.real(nexp[10] * nexp[:]))
chi0_connected = chi0_direct - chi0_disconnected

print(f'\n直接求和: {chi0_direct:.10f}')
print(f'不连接部分: {chi0_disconnected:.10f}')
print(f'连接部分: {chi0_connected:.15f}')