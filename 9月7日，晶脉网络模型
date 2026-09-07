"""
晶脉网络模型.py

晶脉网络是关系本体论下的一个半吊子新模型。
它不是海森堡模型的换名，而是对自由度位置做了根本改变。

核心定义：
    自旋在关系上，不在节点上。
    节点只是关系交汇处。
    两条关系若共享一个节点，则它们相互作用。

与海森堡图模型的区别：
    海森堡图：自旋在节点，边是相互作用
    晶脉网络：自旋在边，节点是关系交汇处

数学形式：
    对原图 G=(V,E)，有 M=|E| 条关系。
    每条关系 e 上有一个自旋 σ_e。
    晶脉网络的哈密顿量：
        H = Σ_{(e,f)∈L(G)} J_ef (σ_e^x σ_f^x + σ_e^y σ_f^y + σ_e^z σ_f^z)
    其中 L(G) 是原图的线图，e 和 f 共享原图节点。

关系观测量：
    E(s) = std{ <σ_e^z σ_f^z>_s }  over all e<f
    这是关系对的关联标准差，而非单自旋期望值。

显现系数：
    dE/ds
    关系协同度对观测角度的响应比例。
    是牛顿第二定律 F=dp/dt 的关系物理版本。

作者：李广好
日期：2026-09-07
"""

import numpy as np
import itertools
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


class JingmaiNetwork:
    """
    晶脉网络模型。
    """

    def __init__(self, edges):
        """
        初始化晶脉网络。
        edges: 原图的边列表 [(u1,v1), (u2,v2), ...]
        """
        self.edges = [tuple(sorted(e)) for e in edges]
        self.n_relations = len(self.edges)

        # 生成线图相互作用：共享原树节点的边对
        self.interactions = []
        for a in range(self.n_relations):
            for b in range(a+1, self.n_relations):
                if len(set(self.edges[a]) & set(self.edges[b])) > 0:
                    self.interactions.append((a, b))

    def build_hamiltonian_sector(self, focus_idx, s):
        """
        构建晶脉网络的哈密顿量（全空间形式，小系统可用）。
        focus_idx: 观测入口
        s: 观测角度
        """
        n = self.n_relations
        dim = 2 ** n
        H = lil_matrix((dim, dim), dtype=float)

        sx = np.array([[0,1],[1,0]], dtype=complex)
        sy = np.array([[0,-1j],[1j,0]], dtype=complex)
        sz = np.array([[1,0],[0,-1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)

        def full(op, site):
            ops = [I2] * n
            ops[site] = op
            r = ops[0]
            for k in range(1, n):
                r = np.kron(r, ops[k])
            return r

        for (a, b) in self.interactions:
            w = s if (a == focus_idx or b == focus_idx) else 1.0
            H += w * (full(sx, a) @ full(sx, b)).real
            H += w * (full(sy, a) @ full(sy, b)).real
            H += w * (full(sz, a) @ full(sz, b)).real

        return H.tocsc()

    def compute_E_single(self, focus_idx, s):
        """
        单态视角：取最低本征态，计算关系观测量 E(s)。
        """
        H = self.build_hamiltonian_sector(focus_idx, s)
        evals, evecs = eigsh(H, k=1, which='SA')
        gs = evecs[:, 0]
        return self._compute_E_from_state(gs)

    def compute_E_manifold(self, focus_idx, s, tol=1e-8):
        """
        流形平均视角：对简并流形中所有基态等权平均后计算 E(s)。
        返回 (E, 简并度, 基态能量, 能隙)
        """
        H = self.build_hamiltonian_sector(focus_idx, s)
        H_dense = H.toarray()
        evals, evecs = np.linalg.eigh(H_dense)
        E0 = evals[0]

        degen_indices = []
        for idx in range(len(evals)):
            if abs(evals[idx] - E0) < tol:
                degen_indices.append(idx)

        degen = len(degen_indices)
        gap = evals[1] - evals[0] if len(evals) > 1 else 0.0

        # 流形平均密度矩阵
        dim = 2 ** self.n_relations
        rho = np.zeros((dim, dim), dtype=complex)
        for idx in degen_indices:
            gs = evecs[:, idx]
            rho += np.outer(gs, gs.conj())
        rho /= degen

        sz = np.array([[1,0],[0,-1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)

        pair_expectations = []
        for a in range(self.n_relations):
            for b in range(a+1, self.n_relations):
                ops = [I2] * self.n_relations
                ops[a] = sz
                ops[b] = sz
                op = ops[0]
                for k in range(1, self.n_relations):
                    op = np.kron(op, ops[k])
                exp_val = np.real(np.trace(rho @ op))
                pair_expectations.append(exp_val)

        E_val = float(np.std(pair_expectations))
        return E_val, degen, float(E0), float(gap)

    def _compute_E_from_state(self, state_vector):
        """从状态向量计算关系观测量 E = std{<σ_a^z σ_b^z>}"""
        sz = np.array([[1,0],[0,-1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)

        pair_expectations = []
        for a in range(self.n_relations):
            for b in range(a+1, self.n_relations):
                ops = [I2] * self.n_relations
                ops[a] = sz
                ops[b] = sz
                op = ops[0]
                for k in range(1, self.n_relations):
                    op = np.kron(op, ops[k])
                exp_val = np.real(state_vector.conj().T @ op @ state_vector)
                pair_expectations.append(exp_val)

        return float(np.std(pair_expectations))

    def scan_all_entrances(self, s_values, method='manifold'):
        """
        对所有入口扫描，计算 E(s) 和显现系数。
        method: 'single' 或 'manifold'
        """
        all_results = {}
        for focus_idx in range(self.n_relations):
            E_curve = np.zeros(len(s_values))
            degen_curve = np.zeros(len(s_values), dtype=int)

            for i, s in enumerate(s_values):
                if method == 'single':
                    E_curve[i] = self.compute_E_single(focus_idx, s)
                    degen_curve[i] = 1
                else:
                    E_curve[i], degen_curve[i], _, _ = self.compute_E_manifold(
                        focus_idx, s
                    )

            manifest = np.gradient(E_curve, s_values)

            all_results[str(focus_idx)] = {
                'focus_relation': focus_idx,
                'edge': self.edges[focus_idx],
                'E_curve': E_curve.tolist(),
                'manifest_coefficient': manifest.tolist(),
                'degeneracy': degen_curve.tolist()
            }

        return all_results


if __name__ == "__main__":
    # 测试：树图晶脉网络
    edges = [(0,1),(0,2),(0,3),(3,4),(3,5)]
    model = JingmaiNetwork(edges)

    print(f"晶脉网络：{model.n_relations} 条关系")
    print(f"线图相互作用：{model.interactions}")

    # 单点测试
    E_single = model.compute_E_single(2, 1.0)
    E_manifold, degen, E0, gap = model.compute_E_manifold(2, 1.0)

    print(f"\n入口 2，s=1.0：")
    print(f"  单态 E = {E_single:.6f}")
    print(f"  流形平均 E = {E_manifold:.6f}")
    print(f"  简并度 = {degen}")
    print(f"  基态能量 = {E0:.6f}")
    print(f"  能隙 = {gap:.6f}")
