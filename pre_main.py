from config import parser
import numpy as np
from typing import List, Tuple, Optional
import pandas as pd
from scipy import sparse as sp
import pickle

def load_nodes_edges(node_csv: str, edge_csv: str):
    """
    读取 node/edge，产出:
      nodes_xy: [N,2] (lng,lat)
      mapping 连续化（若节点ID不连续）
      s,e 无向边数组
    """
    node_df = pd.read_csv(node_csv)
    edge_df = pd.read_csv(edge_csv)

    # 连续化节点ID
    node_df = node_df.sort_values("node").reset_index(drop=True)
    node_ids = node_df["node"].to_numpy()
    id_min, id_max = int(node_ids.min()), int(node_ids.max())
    contiguous = (id_min == 0 and id_max == len(node_df) - 1)
    if not contiguous:
        remap = {old: i for i, old in enumerate(node_ids)}
        node_df["node"] = node_df["node"].map(remap)
        if "s_node" in edge_df.columns and "e_node" in edge_df.columns:
            edge_df["s_node"] = edge_df["s_node"].map(remap)
            edge_df["e_node"]  = edge_df["e_node"].map(remap)

    nodes_xy = node_df[["lng", "lat"]].to_numpy(np.float32)  # [N,2]
    N = len(nodes_xy)

    s = edge_df["s_node"].astype(int).to_numpy()
    e = edge_df["e_node"].astype(int).to_numpy()
    mask = (s >= 0) & (s < N) & (e >= 0) & (e < N)
    s, e = s[mask], e[mask]

    return nodes_xy, s, e, N


def build_trajectories_from_tdrive(tdrive_pkl: str) -> List[np.ndarray]:
    """
    tdrive.pkl: DataFrame with columns ['id','timestamp','trajectory'].
    返回 List[np.ndarray[L,3]]: 每条为 (lng,lat,time_seconds)
    处理:
      - 对应 traj_id 的 time_drop 过滤
      - 连续重复点去除
      - 长度<2 的轨迹丢弃
    """
    with open(tdrive_pkl, "rb") as f:
        df = pd.read_pickle(f)

    trajs = []
    bad = 0
    for _, row in df.iterrows():
        # tid = int(row["id"])
        ts_list = list(row["timestamp"])
        xy_list = [list(p) for p in row["trajectory"]]

        # 对齐长度
        if len(ts_list) != len(xy_list):
            L = min(len(ts_list), len(xy_list))
            ts_list = ts_list[:L]
            xy_list = xy_list[:L]

        # 连续重复点去除
        out_xy = [xy_list[0]]
        out_ts = [ts_list[0]]
        for i in range(1, len(ts_list)):
            if xy_list[i] != xy_list[i - 1]:
                out_xy.append(xy_list[i])
                out_ts.append(ts_list[i])

        if len(out_ts) < 2:
            bad += 1
            continue

        arr = np.concatenate(
            [np.array(out_xy, np.float32), np.array(out_ts, np.float32)[:, None]],
            axis=1
        )  # [L,3]
        trajs.append(arr)

    print(f"[DATA] usable trajectories: {len(trajs)}, dropped: {bad}")
    with open(args.trajs, "wb") as f:
        pickle.dump(trajs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return trajs

def build_sparse_A_hat(N: int, s: np.ndarray, e: np.ndarray) -> sp.coo_matrix:
    """
    构建 A+I 并做对称归一化，返回 COO 稀疏矩阵 A_hat
    """
    data = np.ones_like(s, dtype=np.float32)
    A = sp.coo_matrix((data, (s, e)), shape=(N, N), dtype=np.float32)
    A = A + A.T
    A.setdiag(1.0)  # A + I

    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-8), -0.5).astype(np.float32)
    D_inv = sp.diags(deg_inv_sqrt)
    A_hat = D_inv @ A @ D_inv  # 仍稀疏
    with open(args.A_hat, "wb") as f:
        pickle.dump(A_hat.tocoo(), f, protocol=pickle.HIGHEST_PROTOCOL)
    return A_hat.tocoo()


if __name__ == '__main__':
    args = parser.parse_args()

    nodes_xy, s, e, N = load_nodes_edges(args.node_csv, args.edge_csv)
    build_trajectories_from_tdrive(args.tdrive_pkl)
    build_sparse_A_hat(N, s, e)