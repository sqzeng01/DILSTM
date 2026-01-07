import random
from typing import List, Tuple, Optional
import numpy as np
import torch
#数据初始化工具
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def pad_collate(batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    安全 padding：用“最后一个有效点”重复填充，避免 NaN 传入 KDTree。
    input: batch = [np.ndarray[L,3](lon,lat,time_seconds), ...]
    return:
      gps:  [B, Lmax, 3]
      mask: [B, Lmax] (1 有效, 0 填充)
    """
    Lmax = max(x.shape[0] for x in batch)
    B = len(batch)
    gps = np.zeros((B, Lmax, 3), dtype=np.float32)
    mask = np.zeros((B, Lmax), dtype=np.float32)
    for i, arr in enumerate(batch):
        L = arr.shape[0]
        gps[i, :L] = arr
        mask[i, :L] = 1.0
        if L < Lmax:
            gps[i, L:] = arr[L - 1]  # 重复末点
    return gps, mask

def sinusoid_time_encoding(t: torch.Tensor, base_periods=(60., 3600., 86400.)) -> torch.Tensor:
    """
    周期性时间编码: 对秒级时间戳做多个周期(s/min/hour/day)的正余弦映射
    t: [B, L] 或 [L]
    return: [..., 2 * len(base_periods)]
    """
    feats = []
    for P in base_periods:
        angle = 2 * math.pi * t / P
        feats.append(torch.sin(angle))
        feats.append(torch.cos(angle))
    return torch.stack(feats, dim=-1)  # [..., 2K]


#测试工具

import math
from typing import List, Tuple, Optional
import numpy as np
import torch

def split_odd_even(traj: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    odd  = traj[::2]
    even = traj[1::2]
    if len(odd) < 2 or len(even) < 2:
        return None, None
    return odd, even

def build_query_db(trajectories: List[np.ndarray]):
    Q, D, gold = [], [], []
    for T in trajectories:
        a, b = split_odd_even(T)
        if a is None:
            continue
        Q.append(a)
        D.append(b)
        gold.append(len(D)-1)
    return Q, D, np.array(gold, dtype=np.int64)

@torch.no_grad()
def embed_list(model, trajs, device, batch_size=128):
    Z = []
    for i in range(0, len(trajs), batch_size):
        gps_np, mask_np = pad_collate(trajs[i:i+batch_size])
        gps  = torch.from_numpy(gps_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        z = model.embed(gps, mask)  # [B,D]
        Z.append(z.cpu())
    return torch.cat(Z, dim=0).numpy() if Z else np.zeros((0, 1), dtype=np.float32)

def rank_metrics(Zq: np.ndarray, Zd: np.ndarray, gold: np.ndarray, ks=(1,5,10)):
    S = Zq @ Zd.T  # 余弦等价（z 已 L2）
    ranks = []
    hits = {k:0 for k in ks}
    ap_list = []
    for i in range(S.shape[0]):
        order = np.argsort(-S[i])
        # 找 gold 的排名（从1开始）
        where = np.where(order == gold[i])[0]
        if where.size == 0:
            continue
        r = int(where[0]) + 1
        ranks.append(r)
        for k in ks:
            hits[k] += (r <= k)
        ap_list.append(1.0 / r)  # 单一正样本 → AP=1/r
    N = max(1, len(ranks))
    mr   = float(np.mean(ranks))
    mrr  = float(np.mean([1.0/r for r in ranks]))
    mAP  = float(np.mean(ap_list))
    recall = {k: hits[k]/N for k in ks}
    return {"MR": mr, "MRR": mrr, "mAP": mAP, **{f"R@{k}": recall[k] for k in ks}}

def evaluate_retrieval(model, trajectories_test, device, db_sizes=(1000,10000,100000)):
    Q, D, gold = build_query_db(trajectories_test)
    print(f"[Eval] queries={len(Q)} database={len(D)}")
    if len(Q) == 0 or len(D) == 0:
        print("[Eval] 数据不足，跳过评测"); return
    Zq_full = embed_list(model, Q, device)
    Zd_full = embed_list(model, D, device)
    Nd = Zd_full.shape[0]
    rng = np.random.default_rng(0)
    for M in sorted(set(min(M, Nd) for M in db_sizes)):
        sel = np.sort(rng.choice(Nd, size=M, replace=False))
        sel_set = set(sel.tolist())
        keep = [i for i, g in enumerate(gold) if int(g) in sel_set]
        if not keep:
            print(f"[Eval] |D|={M}: 无有效查询，跳过")
            continue
        pos_map = {int(sel[j]): j for j in range(M)}
        Zq = Zq_full[keep]
        Zd = Zd_full[sel]
        gold_sub = np.array([pos_map[int(gold[i])] for i in keep], dtype=np.int64)
        metrics = rank_metrics(Zq, Zd, gold_sub, ks=(1,5,10))
        print(f"[Eval] |D|={M}: {metrics}")


def subsample_traj(traj, keep_ratio):
    L = len(traj); keep = max(2, int(math.ceil(L*keep_ratio)))
    idx = np.sort(np.random.choice(L, keep, replace=False))
    return traj[idx]

def jitter_traj(traj, sigma_deg):  # 经纬度上小抖动（度）
    out = traj.copy()
    noise = np.random.normal(0, sigma_deg, size=out[:, :2].shape).astype(np.float32)
    out[:, :2] = out[:, :2] + noise
    return out

def evaluate_robustness(model, trajectories_test, device,
                        keep_ratios=(0.5,0.7,0.9), sigmas_deg=(1e-5,2e-5,5e-5)):
    Q0, D0, gold0 = build_query_db(trajectories_test)
    if len(Q0) == 0:
        print("[Robust] 数据不足，跳过鲁棒性评测"); return
    Zd0 = embed_list(model, D0, device)
    # baseline
    Zq0 = embed_list(model, Q0, device)
    base = rank_metrics(Zq0, Zd0, gold0, ks=(1,5,10))
    print("[Robust] baseline:", base)
    # 下采样
    for r in keep_ratios:
        Q = [subsample_traj(t, r) for t in Q0]
        Zq = embed_list(model, Q, device)
        m = rank_metrics(Zq, Zd0, gold0, ks=(1,5,10))
        print(f"[Robust] subsample r={r}:", m)
    # 噪声
    for s in sigmas_deg:
        Q = [jitter_traj(t, s) for t in Q0]
        Zq = embed_list(model, Q, device)
        m = rank_metrics(Zq, Zd0, gold0, ks=(1,5,10))
        print(f"[Robust] jitter σ(deg)={s}:", m)


