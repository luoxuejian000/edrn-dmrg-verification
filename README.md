# EDRN DMRG Verification

## 目的

本仓库用于独立验证一维Hubbard模型在临界区（U/t=4，半满）的标度行为。

**核心问题**：中心点的密度-密度关联函数 χ(0) = Σⱼ ⟨n_i₀ n_j⟩ 随链长 L 的标度指数 α 是多少？

**理论预言**（来自关系本体论框架）：**α = 1.75**

**理论论文初稿**：见本仓库 [paper_draft_v1.pdf](./paper_draft_v1.pdf)

## 如何参与验证

### 方法一：使用我们提供的参考脚本
1. 安装TeNPy：`pip install tenpy`
2. 下载本仓库中的 `dmrg_chi0.py`
3. 运行：`python dmrg_chi0.py`
4. 将生成的 `chi0_vs_L.csv` 提交到本仓库的 `results/` 目录（通过 Pull Request 或 Issue）

### 方法二：使用你自己的工具
任何能计算一维Hubbard模型基态关联函数的方法均可：
- DMRG（ITensor, TeNPy, ALPS等）
- 精确对角化（小链长）
- 量子蒙特卡洛
- Bethe ansatz解析计算

**提交格式**：CSV文件，包含列：L, chi0, logL, logChi0

## 数据并排声明

所有验证结果将以**并排放置**而非合并的方式呈现。每个贡献者的数据保留其原始形式和命名，不做统一标准化。贡献者可以选择：
- 列为共同作者（需完成全套8个链长的计算，精度达标）
- 在致谢中被提及
- 匿名贡献

## 当前状态

| 贡献者 | 方法 | 状态 |
|--------|------|------|
| 李广好 (luoxuejian000) | TeNPy DMRG | 计算中 |
| [您的名字] | [您的方法] | 待加入 |

## 论文草稿

本验证对应的理论论文初稿已上传至本仓库：[paper_draft_v1.pdf](./paper_draft_v1.pdf)

## 联系方式

- 仓库维护者：李广好
- 邮箱：448798112@qq.com
- 欢迎在本仓库Issues中讨论技术问题
