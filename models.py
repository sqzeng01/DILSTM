# models.py
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy import sparse as sp
from sklearn.neighbors import KDTree

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sinusoid_time_encoding
import random

class SimpleGCN_Sparse(nn.Module):
    """
    稀疏 A_hat 的两层 GCN:
      Z = Linear2( A_hat · Dropout( ReLU( Linear1( A_hat · X ) ) ) )
    A_hat 为 buffer (sparse)，随 .to(cuda) 迁移
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, A_hat_sparse: torch.Tensor, dropout=0.1):
        super().__init__()
        self.register_buffer("A_hat", A_hat_sparse.coalesce())
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [N, in_dim] (dense)
        H = torch.sparse.mm(self.A_hat, X)
        H = F.relu(self.lin1(H))
        H = self.drop(H)
        H = torch.sparse.mm(self.A_hat, H)
        Z = self.lin2(H)
        return Z


class SoftMatcher(nn.Module):
    """
    可微软配层（候选集合由 KDTree 选取，不可微；候选集合内的权重为可学习可微）:
      logits_{t,k} = α * ( - dist_{t,k} / τ ) + β * < W_q q_t , W_k e_{v_k} >/√d
      P = softmax(logits)
      gps_struct_emb = Σ_k P_{t,k} · e_{v_k}
    """
    def __init__(self, nodes_xy: np.ndarray, K: int = 8, temperature_deg: float = 0.0015,
                 node_emb_dim: int = 64, d_att: int = 32):
        super().__init__()
        self.nodes_xy = nodes_xy.astype(np.float32)  # [N,2], deg
        self.tree = KDTree(self.nodes_xy)            # 欧式度量在小范围近似OK
        self.temperature = nn.Parameter(torch.tensor(float(temperature_deg)))  # 可学习温度
        self.K = K

        # 可学习的候选内打分支路（让 P 真正可微可学）
        self.d_att = int(d_att)
        self.q_proj = nn.Linear(2, self.d_att)                 # [lon,lat] -> d
        self.k_proj = nn.Linear(int(node_emb_dim), self.d_att) # node_emb -> d
        self.alpha = nn.Parameter(torch.tensor(1.0))           # 距离项权重
        self.beta  = nn.Parameter(torch.tensor(1.0))           # 语义项权重

    @torch.no_grad()
    def query_candidates_masked(self, gps_np: np.ndarray, mask_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        只对 mask=1 的位置查询，mask=0 的位置用占位值。
        gps_np:  [B,L,2]
        mask_np: [B,L]
        return:
          idx:  [B,L,K] int64
          dist: [B,L,K] float32
        """
        B, L, _ = gps_np.shape
        idx_all = np.zeros((B, L, self.K), dtype=np.int64)
        dist_all = np.full((B, L, self.K), fill_value=1e9, dtype=np.float32)
        for b in range(B):
            valid = np.where(mask_np[b] > 0.5)[0]
            if valid.size == 0:
                continue
            d, i = self.tree.query(gps_np[b, valid], k=self.K)  # [Lv,K]
            dist_all[b, valid] = d.astype(np.float32)
            idx_all[b, valid]  = i.astype(np.int64)
        return idx_all, dist_all

    # ------- 在 SoftMatcher 内 -------

    def _compute_logits(self,
                        gps_xy: torch.Tensor,  # [B,L,2]
                        valid_mask: torch.Tensor,  # [B,L] 可能是 float，需要转.bool()
                        node_emb: torch.Tensor,  # [N,D]
                        cand_idx: torch.Tensor,  # [B,L,K]
                        cand_dist: torch.Tensor  # [B,L,K]
                        ) -> torch.Tensor:
        # <- 新增：确保是 bool 掩码
        valid_mask = valid_mask.to(torch.bool)

        # 距离项
        logits_dist = -cand_dist / self.temperature.clamp_min(1e-6)  # [B,L,K]

        # 语义注意项
        Q = self.q_proj(gps_xy)  # [B,L,d]
        cand_node_emb = node_emb[cand_idx]  # [B,L,K,D]
        Kproj = self.k_proj(cand_node_emb)  # [B,L,K,d]
        att = (Q.unsqueeze(2) * Kproj).sum(-1) / (self.d_att ** 0.5)  # [B,L,K]

        logits = self.alpha * logits_dist + self.beta * att  # [B,L,K]

        # <- 用 bool 掩码做屏蔽
        logits = logits.masked_fill(~valid_mask.unsqueeze(-1), -1e9)
        return logits

    def forward(self,
                gps_xy: torch.Tensor,  # [B,L,2]
                valid_mask: torch.Tensor,  # [B,L] 可能是 float，需要转.bool()
                node_emb: torch.Tensor,  # [N,D]
                cand_idx: Optional[torch.Tensor] = None,
                cand_dist: Optional[torch.Tensor] = None):

        # <- 新增：入口就转成 bool，后续一致
        valid_mask = valid_mask.to(torch.bool)

        device = node_emb.device
        B, L, _ = gps_xy.shape

        if cand_idx is None or cand_dist is None:
            gps_np = gps_xy.detach().cpu().numpy()
            mask_np = valid_mask.detach().cpu().numpy().astype(np.bool_)
            idx_np, dist_np = self.query_candidates_masked(gps_np, mask_np)
            cand_idx = torch.from_numpy(idx_np).to(device=device, dtype=torch.long)  # [B,L,K]
            cand_dist = torch.from_numpy(dist_np).to(device=device, dtype=torch.float32)  # [B,L,K]
        else:
            cand_idx = cand_idx.to(device=device, dtype=torch.long)
            cand_dist = cand_dist.to(device=device, dtype=torch.float32)

        logits = self._compute_logits(gps_xy, valid_mask, node_emb, cand_idx, cand_dist)  # [B,L,K]
        P = torch.softmax(logits, dim=-1)  # [B,L,K]

        cand_node_emb = node_emb[cand_idx]  # [B,L,K,D]
        gps_struct_emb = torch.sum(P.unsqueeze(-1) * cand_node_emb, dim=-2)  # [B,L,D]
        gps_struct_emb = gps_struct_emb * valid_mask.unsqueeze(-1)
        return gps_struct_emb, P, cand_idx


class HardMatcher(nn.Module):
    """
    硬匹配层：KDTree 取最近邻候选，直接选 Top-1 节点嵌入（等价于 one-hot 权重）。
    """
    def __init__(self, nodes_xy: np.ndarray, K: int = 8):
        super().__init__()
        self.nodes_xy = nodes_xy.astype(np.float32)
        self.tree = KDTree(self.nodes_xy)
        self.K = K

    @torch.no_grad()
    def query_candidates_masked(self, gps_np: np.ndarray, mask_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        B, L, _ = gps_np.shape
        idx_all = np.zeros((B, L, self.K), dtype=np.int64)
        dist_all = np.full((B, L, self.K), fill_value=1e9, dtype=np.float32)
        for b in range(B):
            valid = np.where(mask_np[b] > 0.5)[0]
            if valid.size == 0:
                continue
            d, i = self.tree.query(gps_np[b, valid], k=self.K)
            dist_all[b, valid] = d.astype(np.float32)
            idx_all[b, valid]  = i.astype(np.int64)
        return idx_all, dist_all

    def forward(self,
                gps_xy: torch.Tensor,     # [B,L,2]
                valid_mask: torch.Tensor, # [B,L]
                node_emb: torch.Tensor,   # [N,D]
                cand_idx: Optional[torch.Tensor] = None,
                cand_dist: Optional[torch.Tensor] = None):
        device = node_emb.device
        B, L, _ = gps_xy.shape
        if cand_idx is None or cand_dist is None:
            gps_np  = gps_xy.detach().cpu().numpy()
            mask_np = valid_mask.detach().cpu().numpy()
            idx_np, dist_np = self.query_candidates_masked(gps_np, mask_np)
            cand_idx  = torch.from_numpy(idx_np).to(device)   # [B,L,K]
            cand_dist = torch.from_numpy(dist_np).to(device)  # [B,L,K]
        else:
            cand_idx  = cand_idx.to(device)
            cand_dist = cand_dist.to(device)

        # 选 Top-1（最小距离）
        top1 = cand_dist.argmin(dim=-1, keepdim=True)          # [B,L,1]
        top_nodes = torch.gather(cand_idx, -1, top1).squeeze(-1)  # [B,L]
        gps_struct_emb = node_emb[top_nodes]                   # [B,L,D]
        gps_struct_emb = gps_struct_emb * valid_mask.unsqueeze(-1)
        # one-hot 权重（可用于可视化/对比）
        K = cand_idx.size(-1)
        P_hard = torch.zeros_like(cand_dist).scatter_(-1, top1, 1.0)  # [B,L,K]
        return gps_struct_emb, P_hard, cand_idx


class ResidualLSTM(nn.Module):
    """
    多层 LSTM + 残差 + masked mean-pool
    """
    def __init__(self, d_in: int, d_hid: int, n_layers: int = 2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hid, num_layers=n_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.proj = nn.Linear(d_hid, d_in)
        self.head = nn.Linear(d_in, d_hid)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        h, _ = self.lstm(x)      # [B,L,D_hid]
        h_proj = self.proj(h)    # [B,L,D_in]
        h_res = h_proj + residual
        m = mask.unsqueeze(-1)   # [B,L,1]
        h_res = h_res * m
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = h_res.sum(dim=1) / denom
        z = self.head(pooled)    # [B,D_hid]
        z = F.normalize(z, p=2, dim=-1)
        return z


@dataclass
class AugConfig:
    # 旧增强（保留以兼容）
    gps_noise_sigma: float = 0.0005   # lon/lat 噪声(度)
    time_jitter_sigma: float = 2.0    # 秒
    subsample_keep_ratio: float = 0.85
    graph_lambda: float = 0.15        # 候选域内的 P 平滑强度
    # 额外增强
    crop_min_ratio: float = 0.6       # 随机裁剪的最小比例
    reverse_prob: float = 0.5         # 裁剪后反转概率
    time_warp_sigma: float = 0.05     # 时间扭曲强度(相对比例)
    # --- TrajCL 四类增强参数 ---
    shift_max_m: float = 100.0        # ρm，最大平移（米）
    mask_ratio: float = 0.30          # ρd，点掩蔽比例
    trunc_keep_ratio: float = 0.70    # ρb，截断后保留比例
    dp_epsilon_m: float = 100.0       # ρp，DP 简化阈值（米）


# ====== TrajCL 四种增强 ======
def _meters_to_deg(m: float) -> float:
    # 近似换算：1度 ≈ 111_111 米
    return m / 111_111.0

def aug_point_shift(gps: torch.Tensor, mask: torch.Tensor, cfg: AugConfig):
    """
    点平移（bounded Gaussian）：对有效 (lon,lat) 加噪，截断到 ±ρm。
    """
    B, L, _ = gps.shape
    out = gps.clone()
    rho_deg = _meters_to_deg(cfg.shift_max_m)
    # 正态噪声，取 std = ρm/3，并裁剪
    std = rho_deg / 3.0
    noise = torch.randn(B, L, 2, device=gps.device) * std
    noise = torch.clamp(noise, min=-rho_deg, max=rho_deg)
    out[..., :2] = out[..., :2] + noise * mask.unsqueeze(-1)
    return out, mask.clone()

def aug_point_mask(gps: torch.Tensor, mask: torch.Tensor, cfg: AugConfig):
    """
    点掩蔽：随机将 ρd 比例的有效点置 0（保留至少 1 个点）。
    """
    B, L, _ = gps.shape
    m = mask.clone()
    for b in range(B):
        valid = torch.where(mask[b] > 0.5)[0]
        if valid.numel() < 2:
            continue
        k = max(1, int(valid.numel() * cfg.mask_ratio))
        drop_idx = valid[torch.randperm(valid.numel(), device=gps.device)[:k]]
        m[b, drop_idx] = 0.0
        # 至少保留一个
        if m[b].sum() < 1:
            m[b, valid[0]] = 1.0
    return gps.clone(), m

def aug_truncate(gps: torch.Tensor, mask: torch.Tensor, cfg: AugConfig):
    """
    轨迹截断：随机保留连续子段，占比 ρb。
    """
    B, L, C = gps.shape
    out = torch.zeros_like(gps)
    m = torch.zeros_like(mask)
    for b in range(B):
        valid = torch.where(mask[b] > 0.5)[0]
        if valid.numel() < 2:
            out[b] = gps[b]
            m[b] = mask[b]
            continue
        lo, hi = int(valid[0].item()), int(valid[-1].item())
        length = hi - lo + 1
        keep = max(2, int(length * cfg.trunc_keep_ratio))
        start = torch.randint(lo, hi - keep + 2, (1,), device=gps.device).item()
        end = start + keep
        seg = gps[b, start:end].clone()
        out[b, :keep] = seg
        m[b, :keep] = 1.0
        if keep < L:
            out[b, keep:] = seg[-1]
    return out, m

# ========== 增强池 + 双视图采样 ==========
def make_two_views(gps: torch.Tensor, mask: torch.Tensor, cfg: AugConfig):
    """
    视图1：强制从 {掩蔽, 截断} 中采样一种（论文主增强）
    视图2：从增强池随机采样一种（含 ρm/ρp/裁剪反转/轻度时间扭曲/也允许再来一次掩蔽或截断）
    返回：((gps1, mask1), (gps2, mask2))
    """
    # 第一视图：论文强调的两种主增强（二选一）
    view1_choices = [aug_point_mask, aug_truncate]
    aug1 = random.choice(view1_choices)
    gps1, mask1 = aug1(gps, mask, cfg)

    # 第二视图：增强池（TrajCL 其余两类 + 论文里的两类补充）
    pool = [
        aug_point_shift,     # ρm：小幅位移（米→度近似）
        aug_simplify_dp,     # ρp：Douglas–Peucker 简化
        random_crop_reverse, # 裁剪 + 概率反转
        time_warp_mask,      # 轻度时间扭曲 + 少量掩蔽
        aug_point_mask,      # 再来一次点掩蔽（提高难度）
        aug_truncate,        # 或者再截断一次（与视图1不同随机性）
    ]
    aug2 = random.choice(pool)
    gps2, mask2 = aug2(gps, mask, cfg)

    return (gps1, mask1), (gps2, mask2)


def _point_line_distance(p, a, b):
    # p,a,b: (2,) numpy
    ap = p - a; ab = b - a
    denom = (ab**2).sum()
    if denom < 1e-12:
        return np.linalg.norm(ap)
    t = np.clip((ap @ ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))

def _dp_indices(xy: np.ndarray, eps: float) -> np.ndarray:
    # 返回需保留的索引（bool mask），Douglas–Peucker
    L = xy.shape[0]
    keep = np.zeros(L, dtype=bool)
    keep[0] = True; keep[-1] = True
    stack = [(0, L-1)]
    while stack:
        s, e = stack.pop()
        if e - s <= 1:
            continue
        a, b = xy[s], xy[e]
        max_d, max_i = -1.0, -1
        for i in range(s+1, e):
            d = _point_line_distance(xy[i], a, b)
            if d > max_d:
                max_d, max_i = d, i
        if max_d > eps:
            keep[max_i] = True
            stack.append((s, max_i))
            stack.append((max_i, e))
    # 保证端点
    keep[0] = True; keep[-1] = True
    return keep

def aug_simplify_dp(gps: torch.Tensor, mask: torch.Tensor, cfg: AugConfig):
    """
    轨迹简化（Douglas–Peucker）：阈值 ρp。
    """
    B, L, C = gps.shape
    out = torch.zeros_like(gps)
    m = torch.zeros_like(mask)
    eps_deg = _meters_to_deg(cfg.dp_epsilon_m)
    gps_cpu = gps.detach().cpu().numpy()
    mask_cpu = mask.detach().cpu().numpy()
    for b in range(B):
        valid = np.where(mask_cpu[b] > 0.5)[0]
        if valid.size < 2:
            out[b] = gps[b]
            m[b] = mask[b]
            continue
        xy = gps_cpu[b, valid, :2]
        keep_mask = _dp_indices(xy, eps_deg)
        sel = valid[keep_mask]
        seg = gps[b, sel].to(gps.device)
        keep = seg.size(0)
        out[b, :keep] = seg
        m[b, :keep] = 1.0
        if keep < L:
            out[b, keep:] = seg[-1]
    return out, m


# ====== 旧增强（保留）======
def gps_level_augment(gps: torch.Tensor, mask: torch.Tensor, cfg: AugConfig):
    """
    (lon,lat,time) 加噪 + 下采样
    return: gps_aug, mask_aug
    """
    B, L, _ = gps.shape
    gps_aug = gps.clone()
    noise_xy = torch.randn(B, L, 2, device=gps.device) * cfg.gps_noise_sigma
    noise_t  = torch.randn(B, L, 1, device=gps.device) * cfg.time_jitter_sigma
    gps_aug[..., :2] = gps_aug[..., :2] + noise_xy
    gps_aug[...,  2:] = gps_aug[...,  2:] + noise_t

    mask_aug = mask.clone()
    drop = (torch.rand(B, L, device=gps.device) > cfg.subsample_keep_ratio).float()
    mask_aug = mask_aug * (1.0 - drop)
    need_fix = (mask_aug.sum(dim=1) < 1.0)
    if need_fix.any():
        for b in torch.where(need_fix)[0]:
            first_valid = int((mask[b] > 0).nonzero(as_tuple=True)[0][0].item())
            mask_aug[b, first_valid] = 1.0
    return gps_aug, mask_aug


def graph_level_augment(P: torch.Tensor, cfg: AugConfig) -> torch.Tensor:
    """
    候选域内简单平滑近似:
      P' = (1-λ) P + λ * mean_k(P)
    """
    if cfg.graph_lambda <= 0:
        return P
    P_mean = P.mean(dim=-1, keepdim=True)
    return (1 - cfg.graph_lambda) * P + cfg.graph_lambda * P_mean


# ====== 额外两种轨迹视图增强（正样本来源）======
def random_crop_reverse(gps: torch.Tensor, mask: torch.Tensor, cfg: AugConfig):
    """
    在有效段内随机裁剪子序列，并以一定概率反转时间顺序（保持道路连贯性）。
    """
    B, L, _ = gps.shape
    device = gps.device
    out = torch.zeros_like(gps)
    m = torch.zeros_like(mask)

    for bi in range(B):
        valid_idx = torch.where(mask[bi] > 0.5)[0]
        if valid_idx.numel() < 2:
            out[bi] = gps[bi]
            m[bi] = mask[bi]
            continue
        lo, hi = int(valid_idx[0].item()), int(valid_idx[-1].item())
        length = hi - lo + 1
        keep = max(2, int(length * cfg.crop_min_ratio))
        start = torch.randint(lo, hi - keep + 2, (1,), device=device).item()
        end = start + keep
        seg = gps[bi, start:end].clone()
        if torch.rand(1, device=device).item() < cfg.reverse_prob:
            seg = torch.flip(seg, dims=[0])
        out[bi, :keep] = seg
        m[bi, :keep] = 1.0
        # 其余位置补最后一个有效点，保持数值稳定
        if keep < L:
            out[bi, keep:] = seg[-1]
    return out, m


def time_warp_mask(gps: torch.Tensor, mask: torch.Tensor, cfg: AugConfig):
    """
    轻度时间扭曲：对 time 轴做缩放+平移随机仿射，并掩蔽少量时刻。
    """
    B, L, _ = gps.shape
    out = gps.clone()
    m = mask.clone()

    t = gps[..., 2]  # [B, L]
    for bi in range(B):
        valid = (mask[bi] > 0.5)
        if valid.sum() < 2:
            continue

        tb = t[bi, valid]  # 只取有效位置的时间戳

        # 仿射：t' = scale * t + bias
        mean_dt = torch.clamp(tb[1:] - tb[:-1], min=1.0).mean()
        scale = 1.0 + torch.randn((), device=gps.device) * cfg.time_warp_sigma
        bias  = torch.randn((), device=gps.device) * cfg.time_warp_sigma * mean_dt

        # 写回
        out[bi, valid.bool(), 2] = scale * tb + bias

        # 随机掩蔽少量时刻（drop 5% 有效点）
        drop_n = max(1, int(valid.sum().item() * 0.05))
        idx = torch.where(valid)[0]
        perm = torch.randperm(idx.numel(), device=gps.device)
        sel = idx[perm[:drop_n]]
        m[bi, sel] = 0.0

        # 至少保留一个点
        if m[bi].sum() < 1:
            m[bi, idx[0]] = 1.0

    return out, m


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        # 固定温度，避免 τ→0 的塌缩捷径
        self.register_buffer("tau", torch.tensor(float(temperature)))

    def forward(self, z, z_pos):
        sim = z @ z_pos.t()  # [B,B]
        logits = sim / self.tau.clamp_min(1e-6)
        labels = torch.arange(z.size(0), device=z.device)
        return F.cross_entropy(logits, labels)


class DCTL(nn.Module):
    """
    DCTL 主体：GNN(稀疏) + Soft/Hard Matching + 时间融合 + ResidualLSTM + CL
    - matching_mode: 'soft' or 'hard'（训练/推理均生效）
    """
    def __init__(self,
                 nodes_xy: np.ndarray,           # [N,2]
                 A_hat_sp_coo: sp.coo_matrix,    # 稀疏对称归一化邻接 (scipy COO)
                 node_feat_in_dim: int = 4,
                 gnn_hid: int = 64,
                 node_emb_dim: int = 64,
                 K: int = 8,
                 soft_T_deg: float = 0.0015,
                 enc_hid: int = 128,
                 enc_layers: int = 2,
                 dropout: float = 0.1,
                 aug_cfg: AugConfig = AugConfig(),
                 matching_mode: str = "soft"):
        super().__init__()
        self.N = nodes_xy.shape[0]
        self.nodes_xy_np = nodes_xy.astype(np.float32)

        # (1) 稀疏 A_hat -> torch.sparse_coo_tensor，并注册为 buffer
        indices = np.vstack([A_hat_sp_coo.row, A_hat_sp_coo.col])
        A_hat_torch = torch.sparse_coo_tensor(
            indices, A_hat_sp_coo.data, size=A_hat_sp_coo.shape, dtype=torch.float32
        ).coalesce()
        self.register_buffer("A_hat", A_hat_torch)

        # (2) 节点初始特征 X0 = [x,y,degree,1] 注册为 buffer
        deg = np.asarray(
            sp.csr_matrix((np.ones_like(A_hat_sp_coo.data), (A_hat_sp_coo.row, A_hat_sp_coo.col)),
                          shape=A_hat_sp_coo.shape).sum(axis=1)
        ).ravel().astype(np.float32)
        X0 = np.concatenate([self.nodes_xy_np, deg[:, None], np.ones((self.N, 1), np.float32)], axis=1)  # [N,4]
        self.register_buffer("X0", torch.from_numpy(X0))

        # (3) 稀疏GCN
        self.gnn = SimpleGCN_Sparse(node_feat_in_dim, gnn_hid, node_emb_dim, self.A_hat, dropout=dropout)

        # (4) Matching
        self.K = K
        self.matching_mode = matching_mode.lower().strip()
        if self.matching_mode == "hard":
            self.matcher = HardMatcher(self.nodes_xy_np, K=K)
        else:
            # 传入 node_emb_dim，使 SoftMatcher 的可学习相似度支路可用
            self.matcher = SoftMatcher(self.nodes_xy_np, K=K, temperature_deg=soft_T_deg,
                                       node_emb_dim=node_emb_dim, d_att=32)

        # (5) 时间编码融合 + 序列编码
        self.time_dim = 6
        self.fuse_proj = nn.Linear(node_emb_dim + self.time_dim, node_emb_dim)
        self.encoder = ResidualLSTM(d_in=node_emb_dim, d_hid=enc_hid, n_layers=enc_layers, dropout=dropout)

        # (6) 对比学习
        self.cl_loss = InfoNCE(temperature=0.20)
        self.aug_cfg = aug_cfg

    def _apply_matching(self, node_emb, cand_idx, weights, mask):
        """
        将候选权重 weights 应用于候选节点，若为 hard 模式则将 weights one-hot 化。
        """
        if self.matching_mode == "hard":
            # 转 one-hot
            idx = weights.argmax(dim=-1, keepdim=True)
            oh = torch.zeros_like(weights).scatter_(-1, idx, 1.0)
            weights = oh
        cand_node_emb = node_emb[cand_idx]  # [B,L,K,D]
        gps_struct_emb = torch.sum(weights.unsqueeze(-1) * cand_node_emb, dim=-2)  # [B,L,D]
        gps_struct_emb = gps_struct_emb * mask.unsqueeze(-1)
        return gps_struct_emb

    def encode_batch(self,
                     gps: torch.Tensor,          # [B,L,3]
                     mask: torch.Tensor,         # [B,L]
                     cand_idx: Optional[torch.Tensor] = None,
                     cand_dist: Optional[torch.Tensor] = None,
                     P_override: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 节点嵌入
        node_emb = self.gnn(self.X0)    # [N,D]

        # Matching
        gps_xy = gps[..., :2]
        gps_struct_emb, P, cand_idx_new = self.matcher(gps_xy, mask, node_emb, cand_idx, cand_dist)

        # 图层增强替换（若提供）
        if P_override is not None:
            weights = P_override
            gps_struct_emb = self._apply_matching(node_emb, cand_idx_new, weights, mask)

        # 时间编码 + 融合 + 序列编码
        t = gps[..., 2]
        time_enc = sinusoid_time_encoding(t).to(gps.device)
        x = torch.cat([gps_struct_emb, time_enc], dim=-1)  # [B,L,D+6]
        x = self.fuse_proj(x)
        z = self.encoder(x, mask)  # [B,H], L2 normed
        return z, P

    @staticmethod
    def _argmax_nodes(P: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        top1 = P.argmax(dim=-1, keepdim=True)
        return torch.gather(cand_idx, -1, top1).squeeze(-1)  # [B,L]

    @staticmethod
    def _overlap_matrix(P: torch.Tensor, cand_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = P.shape
        nodes = DCTL._argmax_nodes(P, cand_idx)             # [B,L]
        m = mask.bool()
        n1 = nodes.unsqueeze(1).expand(B, B, L)
        n2 = nodes.unsqueeze(0).expand(B, B, L)
        mm = (m.unsqueeze(1) & m.unsqueeze(0))
        same = (n1 == n2) & mm
        denom = mm.sum(dim=-1).clamp_min(1)
        ov = same.sum(dim=-1).float() / denom.float()       # [B,B]
        eye = torch.eye(B, device=P.device, dtype=torch.bool)
        return ov.masked_fill(eye, -1.0)

    @staticmethod
    def _build_graph_view_by_ref(P_anchor: torch.Tensor,
                                 cand_idx_anchor: torch.Tensor,
                                 ref_nodes: torch.Tensor,
                                 mode: str = "pos") -> torch.Tensor:
        """
        参考轨迹的 argmax 节点序列 ref_nodes[B,L]
        - mode='pos': 保留共同候选
        - mode='neg': 屏蔽共同候选
        """
        match = (cand_idx_anchor == ref_nodes.unsqueeze(-1))  # [B,L,K]
        if mode == "pos":
            P_new = torch.where(match, P_anchor, torch.zeros_like(P_anchor))
        else:
            P_new = torch.where(match, torch.zeros_like(P_anchor), P_anchor)
        s = P_new.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return P_new / s

    def forward(self, gps: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        TrajCL 风格训练：
          - 轨迹级：两个正样本视图（点掩蔽 ρd / 截断 ρb），InfoNCE 取均值；
          - Batch-hard 负样本 pairwise；
          - 图级视图：基于候选重叠在 P 上构造正/负视图（对 hard 模式会自动 one-hot）。
        """
        B = gps.size(0)

        # ===== 1) Anchor 编码 =====
        z_anchor, P_anchor = self.encode_batch(gps, mask)  # [B,H], [B,L,K]

        # ===== 2) 两个正样本（Mask & Trunc） =====
        (gps_pos1, mask_pos1), (gps_pos2, mask_pos2) = make_two_views(gps, mask, self.aug_cfg)
        z_pos1, _ = self.encode_batch(gps_pos1, mask_pos1)
        z_pos2, _ = self.encode_batch(gps_pos2, mask_pos2)

        loss_gps = 0.5 * (self.cl_loss(z_anchor, z_pos1) + self.cl_loss(z_anchor, z_pos2))

        # ===== 3) Batch-hard 负样本（轨迹级 pairwise） =====
        with torch.no_grad():
            sim = z_anchor @ z_anchor.t()  # [B,B]
            sim = sim.masked_fill(torch.eye(B, device=gps.device, dtype=torch.bool), -1e9)
            hard_neg_idx = sim.argmax(dim=-1)  # [B]
        z_hard_neg = z_anchor[hard_neg_idx]
        pred_i = (z_anchor * z_pos1).sum(dim=-1)
        pred_j = (z_anchor * z_hard_neg).sum(dim=-1)
        loss_pair = -F.logsigmoid(pred_i - pred_j).mean()

        # ===== 4) 图级正/负（基于候选重叠，构造 P 视图） =====
        with torch.no_grad():
            node_emb = self.gnn(self.X0)
            _, _, cand_idx_anchor = self.matcher(gps[..., :2], mask, node_emb)  # [B,L,K]

            ov = self._overlap_matrix(P_anchor, cand_idx_anchor, mask)          # [B,B]
            pos_idx = torch.argmax(ov, dim=-1)
            ov_for_neg = torch.where(ov < 0, torch.full_like(ov, float('inf')), ov)
            neg_idx = torch.argmin(ov_for_neg, dim=-1)

            ref_nodes_pos = self._argmax_nodes(P_anchor[pos_idx], cand_idx_anchor[pos_idx])  # [B,L]

        P_pos_view = self._build_graph_view_by_ref(P_anchor, cand_idx_anchor, ref_nodes_pos, mode="pos")
        P_neg_view = self._build_graph_view_by_ref(P_anchor, cand_idx_anchor, ref_nodes_pos, mode="neg")
        # 编码时应用 matching_mode（soft/hard 都兼容）
        z_gpos, _ = self.encode_batch(gps, mask, P_override=P_pos_view)
        z_gneg, _ = self.encode_batch(gps, mask, P_override=P_neg_view)

        pred_i_g = (z_anchor * z_gpos).sum(dim=-1)
        pred_j_g = (z_anchor * z_gneg).sum(dim=-1)
        loss_graph = -F.logsigmoid(pred_i_g - pred_j_g).mean()

        # ===== 5) 汇总 =====
        loss = loss_gps + loss_graph + 0.5 * loss_pair

        stats = {
            "loss": float(loss.detach().cpu()),
            "loss_gps": float(loss_gps.detach().cpu()),
            "loss_graph": float(loss_graph.detach().cpu()),
            "loss_pair": float(loss_pair.detach().cpu()),
        }
        return loss, stats

    @torch.no_grad()
    def embed(self, gps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode_batch(gps, mask)
        return z
