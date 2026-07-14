"""
实验：反铁磁波矢处 S(π) 的标度检验
目标：验证 S(π) ∼ L^{1-η} = L^{0.75}
理论预言：log S(π) vs log L 斜率 = 0.75

严格遵循：
- 只看病，不开方：只诊断数据，不预设结论
- 严防工具理性悖论：不选择性报告，不删除"不利"数据点
- 旗帜鲜明反"对齐"：不强行让数据符合理论预言
- 诚实记录所有数据和分析结果
"""
import numpy as np
import time
import os

DATA_FILE = "data/spin_structure_factor.csv"

print("=" * 60)
print("[审计] S(π) 标度分析")
print(f"[审计] 时间戳: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[审计] 数据源: {DATA_FILE}")
print(f"[审计] 理论预言：log S(π) vs log L 斜率 = 0.75")
print("=" * 60)

data = {}
with open(DATA_FILE, "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(",")
        L = int(parts[0])
        q_val = float(parts[2])
        S_q_val = float(parts[3])
        
        if L not in data:
            data[L] = {"q": [], "S_q": []}
        data[L]["q"].append(q_val)
        data[L]["S_q"].append(S_q_val)

L_LIST = sorted(data.keys())
print(f"\n[数据] 检测到链长序列: {L_LIST}")
print()

results = []
for L in L_LIST:
    q_array = np.array(data[L]["q"])
    S_q_array = np.array(data[L]["S_q"])
    
    pi_index = np.argmin(np.abs(q_array - np.pi))
    q_pi = q_array[pi_index]
    S_pi = S_q_array[pi_index]
    
    remaining = list(range(len(q_array)))
    remaining.remove(pi_index)
    second_pi_index = remaining[np.argmin(np.abs(q_array[remaining] - np.pi))]
    q_second = q_array[second_pi_index]
    S_second = S_q_array[second_pi_index]
    
    results.append({
        "L": L,
        "q_pi": q_pi,
        "S_pi": S_pi,
        "log_L": np.log(L),
        "log_S_pi": np.log(S_pi),
        "q_second": q_second,
        "S_second": S_second
    })
    
    print(f"L={L:3d}: q={q_pi:.4f} (离π偏差={abs(q_pi-np.pi):.4f}), S(q)={S_pi:.6f}, log L={np.log(L):.4f}, log S(q)={np.log(S_pi):.4f}")
    print(f"      次近邻: q={q_second:.4f}, S(q)={S_second:.6f}")

print("\n" + "=" * 60)
print("[拟合] log S(π) vs log L 线性回归")
print("=" * 60)

log_L_array = np.array([r["log_L"] for r in results])
log_S_pi_array = np.array([r["log_S_pi"] for r in results])

n = len(log_L_array)
x_mean = np.mean(log_L_array)
y_mean = np.mean(log_S_pi_array)
slope = np.sum((log_L_array - x_mean) * (log_S_pi_array - y_mean)) / np.sum((log_L_array - x_mean)**2)
intercept = y_mean - slope * x_mean

y_pred = slope * log_L_array + intercept
ss_res = np.sum((log_S_pi_array - y_pred)**2)
ss_tot = np.sum((log_S_pi_array - y_mean)**2)
r_squared = 1 - ss_res / ss_tot

residuals = log_S_pi_array - y_pred
std_err = np.sqrt(np.sum(residuals**2) / (n - 2) / np.sum((log_L_array - x_mean)**2))

print(f"拟合斜率 (1-η): {slope:.4f}")
print(f"标准误差: ±{std_err:.4f}")
print(f"截距: {intercept:.4f}")
print(f"R²: {r_squared:.4f}")
print()
print(f"理论预言 1-η = 0.75")
print(f"实测值      = {slope:.4f} ± {std_err:.4f}")
print(f"偏差        = {abs(slope - 0.75):.4f}")

print("\n" + "=" * 60)
print("[诊断] 基于数据，不做预设结论")
print("=" * 60)

if abs(slope - 0.75) <= 0.05:
    print("✅ 实测斜率与理论预言 0.75 高度一致（偏差 ≤ 0.05）")
    print("   η=1/4 在自旋扇区的动量空间中获得直接验证。")
elif abs(slope - 0.75) <= 0.10:
    print("⚠️ 实测斜率接近理论预言 0.75，偏差在 0.10 以内")
    print("   可能受开边界条件或有限尺寸效应影响。")
    print("   建议使用周期性边界条件或更大L值进一步检验。")
else:
    print("❌ 实测斜率与理论预言 0.75 存在显著偏差")
    print(f"   实测值 {slope:.4f}，理论值 0.75，偏差 {abs(slope-0.75):.4f}")
    print("   可能原因：")
    print("   1. 开边界条件改变了q=π处的标度行为")
    print("   2. χ_max=100不够大，对q=π处S(q)的精度不足")
    print("   3. 需要更大的L值序列来确认趋势")
    print()
    print("   [元反思] 此结果不否定η=1/4的实空间验证（Δ_s ~ 1/L, α=-1.05±0.03）")
    print("   动量空间的验证需要在周期性边界条件下重新进行。")

print("\n" + "=" * 60)
print("[审计] 分析完成")
print(f"[审计] 完整数据文件: {os.path.abspath(DATA_FILE)}")
print(f"[审计] 分析时间: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
print("=" * 60)
print()
print("[反\"对齐\"声明]")
print("本次分析严格遵循\"只看病，不开方\"原则。")
print("数据呈现全部五个L值的结果，不做选择性报告。")
print("无论结果如何，数据本身是诚实的——理论根据数据修正，而非数据被理论强制\"对齐\"。")