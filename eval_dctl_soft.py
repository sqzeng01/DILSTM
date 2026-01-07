# baselines/eval_dctl_soft.py
# 你的 DCTL（软匹配）—— 训练 + 三类指标 + 鲁棒性 + 稳定性（已统一切分/评测）
import pickle, numpy as np, time, random
from pathlib import Path
from typing import List, Dict
import torch
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from config import parser
from pre_main import load_nodes_edges
from utils import build_query_db, evaluate_retrieval, evaluate_robustness, pad_collate, embed_list
from trainer import train_loop

# ----------------- 追加/统一参数 -----------------
def _safe_add(*args, **kwargs):
    try: parser.add_argument(*args, **kwargs)
    except: pass

_safe_add("--save_dir", default="saved_models")
_safe_add("--train", default=True)
_safe_add("--hybrid_mode", default="fusion")       # rerank / fusion
_safe_add("--alpha", type=float, default=0.3)      # hybrid 融合权重
_safe_add("--topM", type=int, default=100)         # rerank 候选
_safe_add("--use_tfidf", action="store_true", default=True)
_safe_add("--tau_topo", type=float, default=0.1)   # 统一拓扑阈值名
_safe_add("--stable_rounds", type=int, default=5)
_safe_add("--stable_ratio", type=float, default=0.8)

# 统一切分控制
_safe_add("--seed", type=int, default=0)
_safe_add("--split_ratio", type=float, default=0.9)
_safe_add("--shuffle_split", action="store_true", default=False)

# ----------------- 通用工具 -----------------
def _load_trajs(path, maxn=None):
    with open(path, "rb") as f: T = pickle.load(f)
    if maxn and len(T) > maxn: T = T[:maxn]
    return [np.asarray(t, dtype=np.float32) for t in T]

def _split_train_test(trajs, ratio=0.9, seed=0, shuffle=False):
    idx = np.arange(len(trajs))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    cut = max(1, int(len(trajs)*ratio))
    tr_idx, te_idx = idx[:cut], idx[cut:]
    return [trajs[i] for i in tr_idx], [trajs[i] for i in te_idx]

def _bow_nodes(trajs, nodes_xy) -> csr_matrix:
    tree = cKDTree(nodes_xy[:,:2])
    n, N = len(trajs), nodes_xy.shape[0]
    indptr, indices, data = [0], [], []
    for t in trajs:
        if len(t)==0: indptr.append(indptr[-1]); continue
        ids = tree.query(t[:,:2], k=1, workers=-1)[1]
        u, c = np.unique(ids, return_counts=True)
        indices.extend(u.tolist()); data.extend(c.astype(np.float32).tolist())
        indptr.append(len(indices))
    return csr_matrix((np.array(data,dtype=np.float32), np.array(indices,dtype=np.int32), np.array(indptr,dtype=np.int32)),
                      shape=(n, N), dtype=np.float32)

def _tfidf_l2(S: csr_matrix):
    df = (S>0).astype(np.float32).sum(axis=0).A1 + 1.0
    idf = np.log((S.shape[0]+1.0)/df) + 1.0
    S = S.multiply(idf)
    row = np.sqrt(np.asarray(S.multiply(S).sum(axis=1)).ravel()) + 1e-8
    return S.multiply(1.0/row[:,None]).tocsr()

def _row_l2(S: csr_matrix):
    row = np.sqrt(np.asarray(S.multiply(S).sum(axis=1)).ravel()) + 1e-8
    return S.multiply(1.0/row[:,None]).tocsr()

def _topo_edges(node_ids: np.ndarray):
    es = set()
    for i in range(len(node_ids)-1):
        u, v = int(node_ids[i]), int(node_ids[i+1])
        if u == v: continue
        if u > v: u, v = v, u
        es.add((u, v))
    return es

def _topo_eval(S, EQ: List[set], ED: List[set], gold, tau=0.5):
    positives = []
    for i in range(len(EQ)):
        pos=set()
        for j in range(len(ED)):
            inter=len(EQ[i] & ED[j]); union=len(EQ[i] | ED[j]) or 1
            if inter/union >= tau: pos.add(j)
        positives.append(pos)
    def recall_at_k(k):
        ok=0
        for i in range(S.shape[0]):
            idx = np.argpartition(-S[i], kth=min(k,S.shape[1]-1))[:k]
            if len(positives[i].intersection(set(idx)))>0: ok+=1
        return ok / S.shape[0]
    mt1=0.0
    for i in range(S.shape[0]):
        j=int(np.argmax(S[i]))
        inter=len(EQ[i] & ED[j]); union=len(EQ[i] | ED[j]) or 1
        mt1 += inter/union
    mt1/=S.shape[0]
    def exact_recall_at_k(k):
        ok=0
        for i in range(S.shape[0]):
            idx = np.argpartition(-S[i], kth=min(k,S.shape[1]-1))[:k]
            if gold[i] in idx: ok+=1
        return ok / S.shape[0]

    # 在你生成 positives 之后（或生成前你先算 edges），加这个统计
    pos_sizes = np.array([len(p) for p in positives], dtype=np.int64)
    print("[Topo] positives per query stats:",
          "min", pos_sizes.min(),
          "mean", pos_sizes.mean(),
          "median", np.median(pos_sizes),
          "p90", np.percentile(pos_sizes, 90),
          "zero_ratio", (pos_sizes == 0).mean())

    return {"Topo R@1": recall_at_k(1), "Topo R@5": recall_at_k(5), "Topo R@10": recall_at_k(10),
            "Mean Top-1 Jaccard": mt1,
            "Exact R@1": exact_recall_at_k(1), "Exact R@5": exact_recall_at_k(5), "Exact R@10": exact_recall_at_k(10)}

def _metrics_from_ranks(ranks, ks=(1,5,10)):
    r = np.asarray(ranks, dtype=np.int64)
    out = {"MR": float(r.mean()), "MRR": float(np.mean(1.0/r)), "mAP": float(np.mean(1.0/r))}
    for k in ks: out[f"R@{k}"] = float((r<=k).mean())
    return out

def _ranks_from_scores(S, gold):
    ranks=[]
    for i in range(S.shape[0]):
        g = gold[i]; ranks.append(1 + int(np.sum(S[i] > S[i,g])))
    return ranks

def _stability_eval(Zq, Zd, gold, rounds=5, ratio=0.8, seed=42):
    rng = np.random.RandomState(seed)
    S = Zq @ Zd.T
    Nq = S.shape[0]; k = max(1, int(Nq*ratio))
    stats: Dict[str, list] = {"R@1":[], "R@5":[], "R@10":[], "MR":[], "MRR":[], "mAP":[]}
    for _ in range(rounds):
        idx = rng.choice(Nq, size=k, replace=False)
        ranks = _ranks_from_scores(S[idx], [gold[i] for i in idx])
        res = _metrics_from_ranks(np.asarray(ranks))
        for key in stats: stats[key].append(res.get(key, res.get(key, 0.0)))
    summary = {}
    for k,v in stats.items():
        summary[f"{k}_mean"] = float(np.mean(v))
        summary[f"{k}_std"]  = float(np.std(v))
    return summary

# ----------------- 主流程 -----------------
def main():
    args = parser.parse_args()
    device = args.device
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    data = args.datasets
    print(f"split:{args.split_ratio}")
    # 数据
    nodes_xy, s_idx, e_idx, N = load_nodes_edges(args.node_csv, args.edge_csv)
    with open(args.A_hat, "rb") as f: A_hat = pickle.load(f)
    trajs = _load_trajs(args.trajs, args.max_trajs)
    print(args.split_ratio)
    train_set, test_set = _split_train_test(trajs, ratio=0.95, seed=args.seed, shuffle=args.shuffle_split)
    Q, D, gold = build_query_db(test_set)

    # 训练 DCTL(soft) —— “按原样训练”
    model_path = Path(args.save_dir) / "dctl_soft.pt"
    if args.train or (not model_path.exists()):
        dctl = train_loop(train_set, nodes_xy, A_hat, device=device,
                          epochs=args.epochs, batch_size=args.batch_size,
                          K=args.K, soft_T_deg=args.soft_T_deg, enc_hid=args.enc_hid,
                          trainable=True, path=str(model_path), matching_mode="soft")
    else:
        dctl = train_loop(train_set, nodes_xy, A_hat, device=device,
                          epochs=1, batch_size=args.batch_size,
                          K=args.K, soft_T_deg=args.soft_T_deg, enc_hid=args.enc_hid,
                          trainable=False, path=str(model_path), matching_mode="soft")

    # ========== Ⅰ. IID ==========
    print("\n[DCTL-Soft] IID Retrieval:")
    evaluate_retrieval(dctl, test_set, device)

    with torch.no_grad():
        Zq = embed_list(dctl, Q, device=device, batch_size=args.batch_size)
        Zd = embed_list(dctl, D, device=device, batch_size=args.batch_size)
    S = Zq @ Zd.T
    ranks = _ranks_from_scores(S, gold)
    print("[DCTL-Soft] IID summary:", _metrics_from_ranks(np.asarray(ranks)))

    # ========== Ⅱ. Topology ==========
    print("\n[DCTL-Soft] Topology-aware metrics:")
    tree = cKDTree(nodes_xy[:,:2])
    EQ = [_topo_edges(tree.query(t[:, :2], k=1, workers=-1)[1].astype(np.int64)) for t in Q]
    ED = [_topo_edges(tree.query(t[:, :2], k=1, workers=-1)[1].astype(np.int64)) for t in D]
    # tau = getattr(args, "tau_topo", 0.5)
    tau = 0.1
    topo = _topo_eval(S, EQ, ED, gold, tau=tau)
    for k, v in topo.items():
        print(f"{k}: {v:.4f}")

    # ========== Ⅲ. Hybrid ==========
    print("\n[DCTL-Soft] Hybrid:")
    SQ = _bow_nodes(Q, nodes_xy).tocsr()
    SD = _bow_nodes(D, nodes_xy).tocsr()
    SQ = _tfidf_l2(SQ) if args.use_tfidf else _row_l2(SQ)
    SD = _tfidf_l2(SD) if args.use_tfidf else _row_l2(SD)

    if args.hybrid_mode.lower() == "fusion":
        ranks = []
        for st in range(0, len(Q), 256):
            s_tp = (SQ[st:st+256] @ SD.T).toarray()
            s = args.alpha * S[st:st+256] + (1.0-args.alpha) * s_tp
            for i in range(s.shape[0]):
                g = gold[st+i]
                ranks.append(1 + int(np.sum(s[i] > s[i, g])))
        print(_metrics_from_ranks(np.asarray(ranks)))
    else:
        M = int(args.topM)
        ranks=[]
        for st in range(0, len(Q), 256):
            s_tr = S[st:st+256]
            kth = min(M, s_tr.shape[1]-1)
            idx = np.argpartition(-s_tr, kth=kth, axis=1)[:, :M]
            for i in range(idx.shape[0]):
                cand = idx[i]
                row = SQ.getrow(st+i); cand_mat = SD[cand]
                s_tp = (row @ cand_mat.T).toarray().ravel()
                s = args.alpha * s_tr[i, cand] + (1.0-args.alpha) * s_tp
                g = gold[st+i]
                pos = np.where(cand==g)[0]
                ranks.append(1+int(np.sum(s> s[pos[0]])) if len(pos)>0 else 10**6)
        print(_metrics_from_ranks(np.asarray(ranks)))

    # ========== Ⅳ. 鲁棒性 ==========
    # print("\n[DCTL-Soft] Robustness:")
    # kr = tuple(float(x) for x in args.robust_keep.split(","))
    # sg = tuple(float(x) for x in args.robust_sigma_deg.split(","))
    # evaluate_robustness(dctl, test_set, device, keep_ratios=kr, sigmas_deg=sg)
    #
    # # ========== Ⅴ. 稳定性 ==========
    # print("\n[DCTL-Soft] Stability (bootstrap):")
    # stab = _stability_eval(Zq, Zd, gold, rounds=args.stable_rounds, ratio=args.stable_ratio)
    # for k,v in stab.items(): print(f"{k}: {v:.4f}")



    print("\n[dctl] Hybrid-Robustness:")
    kr = tuple(float(x) for x in args.robust_keep.split(","))
    sg = tuple(float(x) for x in args.robust_sigma_deg.split(","))

    # ① 固定数据库的语义与拓扑（保证对比公平）
    SD = _bow_nodes(D, nodes_xy).tocsr()
    SD = _tfidf_l2(SD) if args.use_tfidf else _row_l2(SD)

    def _hy_metrics(Zq: np.ndarray, SQ: csr_matrix) -> dict:
        """给定查询语义 Zq 与查询拓扑 SQ，计算 Hybrid 的排名指标"""
        S_tr = Zq @ Zd.T
        ranks = []
        if args.hybrid_mode.lower() == "fusion":
            for st in range(0, SQ.shape[0], 256):
                s_tp = (SQ[st:st+256] @ SD.T).toarray()
                s = args.alpha * S_tr[st:st+256] + (1.0 - args.alpha) * s_tp
                for i in range(s.shape[0]):
                    g = gold[st+i]
                    ranks.append(1 + int(np.sum(s[i] > s[i, g])))
        else:  # rerank
            M = int(args.topM)
            for st in range(0, SQ.shape[0], 256):
                s_tr = S_tr[st:st+256]
                kth = min(M, s_tr.shape[1]-1)
                idx = np.argpartition(-s_tr, kth=kth, axis=1)[:,:M]
                for i in range(idx.shape[0]):
                    cand = idx[i]
                    row = SQ.getrow(st+i)
                    cand_mat = SD[cand]
                    s_tp = (row @ cand_mat.T).toarray().ravel()
                    s = args.alpha * s_tr[i, cand] + (1.0 - args.alpha) * s_tp
                    g = gold[st+i]
                    pos = np.where(cand == g)[0]
                    ranks.append(1 + int(np.sum(s > s[pos[0]])) if len(pos)>0 else 10**6)
        return _metrics_from_ranks(ranks, ks=(1,5,10))

    # ② 随机丢点鲁棒性（只扰动 Q，一直固定 D/SD）
    rng = np.random.default_rng(42)
    def _keep_points(traj: np.ndarray, keep_ratio: float) -> np.ndarray:
        L = len(traj); k = max(2, int(np.ceil(L*keep_ratio)))
        if k >= L: return traj.copy()
        idx = np.sort(rng.choice(L, size=k, replace=False))
        return traj[idx]

    for keep in sorted(set(kr)):
        Q_keep = [_keep_points(t, keep) for t in Q]
        with torch.no_grad():
            Zq_keep = embed_list(dctl, Q_keep, device=device, batch_size=args.batch_size)
        SQ_keep = _bow_nodes(Q_keep, nodes_xy).tocsr()
        SQ_keep = _tfidf_l2(SQ_keep) if args.use_tfidf else _row_l2(SQ_keep)
        m = _hy_metrics(Zq_keep, SQ_keep)
        print(f"[drop] keep_ratio={keep}: {m}")

    # ③ 高斯扰动鲁棒性（只扰动 Q，一直固定 D/SD）
    def _gaussian_noise(traj: np.ndarray, sigma_deg: float) -> np.ndarray:
        if len(traj) == 0: return traj.copy()
        out = traj.copy()
        noise = rng.normal(0.0, sigma_deg, size=(len(out), 2)).astype(np.float32)
        out[:, :2] = out[:, :2] + noise
        return out

    for sgma in sorted(set(sg)):
        Q_noise = [_gaussian_noise(t, sgma) for t in Q]
        with torch.no_grad():
            Zq_noise = embed_list(dctl, Q_noise, device=device, batch_size=args.batch_size)
        SQ_noise = _bow_nodes(Q_noise, nodes_xy).tocsr()
        SQ_noise = _tfidf_l2(SQ_noise) if args.use_tfidf else _row_l2(SQ_noise)
        m = _hy_metrics(Zq_noise, SQ_noise)
        print(f"[noise] sigma_deg={sgma}: {m}")

if __name__ == "__main__":
    main()
