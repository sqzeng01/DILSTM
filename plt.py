# relative_change_plots.py
# 仅用 matplotlib；每张图单独 figure；不指定颜色；对比 y / y@x=1（相对比例）

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============== 你的原始数据（keep_ratio=r）================
r_vals = [1.0, 0.9, 0.7, 0.5]
data_mr_r = {
    "e2dtc":        [4.4014, 4.4473, 5.4930, 7.0723],
    "traj2simvec":  [8.4084, 8.3655, 13.5164, 21.2389],
    "trajcl":       [1.7250, 1.9540, 2.4330, 3.0530],
    "gts":          [23.8909, 41.5808, 85.7046, 151.0880],
    "st2":          [1.3529, 1.4237, 1.9140, 3.9080],
    "grlstm":       [20.4485, 20.5368, 26.1691, 36.3103],
    "OURS":         [1.0707, 1.0807, 1.1821, 1.2814],
}
data_hr1_r = {
    "e2dtc":        [0.6822, 0.6566, 0.5853, 0.5023],
    "traj2simvec":  [0.6959, 0.6382, 0.5625, 0.4908],
    "trajcl":       [0.7940, 0.7590, 0.7160, 0.6490],
    "gts":          [0.6760, 0.6216, 0.5444, 0.4777],
    "st2":          [0.9497, 0.9386, 0.9039, 0.8335],
    "grlstm":       [0.5238, 0.5016, 0.4314, 0.3640],
    "OURS":         [0.9691, 0.9591, 0.9426, 0.9141],
}
data_hr10_r = {
    "e2dtc":        [0.9502, 0.9455, 0.9227, 0.8885],
    "traj2simvec":  [0.9205, 0.9066, 0.8597, 0.8005],
    "trajcl":       [0.9830, 0.9760, 0.9630, 0.9580],
    "gts":          [0.9170, 0.9012, 0.8555, 0.7973],
    "st2":          [0.9955, 0.9946, 0.9908, 0.9758],
    "grlstm":       [0.8282, 0.8140, 0.7709, 0.7226],
    "OURS":         [0.9993, 0.9983, 0.9973, 0.9954],
}

# =============== 你的原始数据（σ 扭曲）================
sigma_vals = ["1", "1e-05", "2e-05", "5e-05"]
data_mr_s = {
    "e2dtc":       [4.4014, 4.1416, 4.2628, 4.7706],
    "traj2simvec": [8.4084, 8.6500, 9.4219, 11.8911],
    "trajcl":      [1.7250, 1.7270, 1.9530, 2.0070],
    "gts":         [23.8909, 25.0857, 27.3740, 25.5628],
    "st2":         [1.3529, 1.3770, 1.5362, 2.1220],
    "grlstm":      [20.4485, 20.4400, 21.7636, 23.1433],
    "OURS":        [1.0707, 1.0819, 1.1635, 1.2523],
}
data_hr1_s = {
    "e2dtc":       [0.6822, 0.6833, 0.6750, 0.6415],
    "traj2simvec": [0.6959, 0.6896, 0.6752, 0.6232],
    "trajcl":      [0.7940, 0.7920, 0.7740, 0.7570],
    "gts":         [0.6760, 0.6739, 0.6572, 0.6137],
    "st2":         [0.9497, 0.9454, 0.9392, 0.9201],
    "grlstm":      [0.5238, 0.5260, 0.5179, 0.4846],
    "OURS":        [0.9691, 0.9609, 0.9571, 0.9458],
}
data_hr10_s = {
    "e2dtc":       [0.9502, 0.9503, 0.9497, 0.9404],
    "traj2simvec": [0.9205, 0.9169, 0.9097, 0.8878],
    "trajcl":      [0.9830, 0.9840, 0.9780, 0.9750],
    "gts":         [0.9170, 0.9134, 0.9065, 0.8853],
    "st2":         [0.9955, 0.9950, 0.9948, 0.9903],
    "grlstm":      [0.8282, 0.8264, 0.8264, 0.8052],
    "OURS":        [0.9993, 0.9992, 0.9989, 0.9975],
}

# =============== 相对值（以 x=1 为基准） ===============
def rel_to_baseline(vals, baseline_index=0):
    v = np.asarray(vals, dtype=float)
    return v / v[baseline_index]        # 如需百分比变化：return (v / v[0] - 1.0) * 100.0

# =============== 统一排版（不设颜色） ===============
plt.rcParams.update({
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 300,
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
markers    = ["o", "s", "^", "D", "v", "P", "X"]
linestyles = ["solid", "dashed", "dashdot", (0,(1,1)), (0,(5,1,1,1)), (0,(3,1,1,1,1,1)), "solid"]

def plot_relative_curves(data_dict, x_vals, x_label, title, out_prefix):
    # 保存相对值 CSV
    rel = {m: rel_to_baseline(y).tolist() for m, y in data_dict.items()}
    df = pd.DataFrame(rel, index=x_vals).T
    df.index.name = "Model"
    df.columns = [str(x) for x in x_vals]
    df.to_csv(f"{out_prefix}_relative.csv", encoding="utf-8-sig")

    # 画图
    plt.figure()
    for i, (model, y) in enumerate(data_dict.items()):
        rel_y = rel_to_baseline(y)
        plt.plot(range(len(x_vals)), rel_y,
                 marker=markers[i % len(markers)],
                 linestyle=linestyles[i % len(linestyles)],
                 label=model)
    plt.xticks(range(len(x_vals)), [str(x) for x in x_vals])
    plt.xlabel(x_label)
    plt.ylabel("Relative to baseline (y / y@x=1)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    plt.close()

# ---- keep_ratio（r）相对变化 ----
plot_relative_curves(data_mr_r,  r_vals, "keep_ratio $r$", "Relative Hybrid MR vs. $r$ (baseline $r$=1.0)",   "rel_r_hy_mr")
plot_relative_curves(data_hr1_r, r_vals, "keep_ratio $r$", "Relative Hybrid HR@1 vs. $r$ (baseline $r$=1.0)", "rel_r_hy_hr1")
plot_relative_curves(data_hr10_r,r_vals, "keep_ratio $r$", "Relative Hybrid HR@10 vs. $r$ (baseline $r$=1.0)","rel_r_hy_hr10")

# ---- σ 相对变化 ----
plot_relative_curves(data_mr_s,  sigma_vals, r"$\sigma$", "Relative Hybrid MR vs. $\sigma$ (baseline $\sigma$=1)",   "rel_sigma_hy_mr")
plot_relative_curves(data_hr1_s, sigma_vals, r"$\sigma$", "Relative Hybrid HR@1 vs. $\sigma$ (baseline $\sigma$=1)", "rel_sigma_hy_hr1")
plot_relative_curves(data_hr10_s,sigma_vals, r"$\sigma$", "Relative Hybrid HR@10 vs. $\sigma$ (baseline $\sigma$=1)","rel_sigma_hy_hr10")

print("Saved:",
      "rel_r_hy_mr.[png/pdf/svg]  rel_r_hy_hr1.[png/pdf/svg]  rel_r_hy_hr10.[png/pdf/svg]",
      "rel_sigma_hy_mr.[png/pdf/svg]  rel_sigma_hy_hr1.[png/pdf/svg]  rel_sigma_hy_hr10.[png/pdf/svg]")
