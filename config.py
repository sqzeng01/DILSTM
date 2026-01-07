import argparse
import torch
parser = argparse.ArgumentParser()
# 默认从 /mnt/data 读取（你上传的数据目录）
parser.add_argument("--node_csv", default="dataset/roma/node.csv")
parser.add_argument("--edge_csv", default="dataset/roma/edge.csv")
parser.add_argument("--tdrive_pkl", default="dataset/roma/out.pkl")
parser.add_argument("--A_hat", default="dataset/roma/A_hat.pkl")
parser.add_argument("--trajs", default="dataset/roma/trajs.pkl")
parser.add_argument("--datasets", default="roma")#beijing   roma   shanghai
parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=50)#dctl 150, st2vec 20, gts 1,grlstm
parser.add_argument("--K", type=int, default=10)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--soft_T_deg", type=float, default=0.002)  # 与经纬度量纲一致
parser.add_argument("--enc_hid", type=int, default=128)
parser.add_argument("--max_trajs", type=int, default=None, help="为了加速可限制轨迹数，例如 5000")
# 评测相关开关
parser.add_argument("--eval", action="store_true", help="训练后运行准确性与鲁棒性评测", default=True)
parser.add_argument("--db_sizes", type=str, default="1000,10000,100000", help="检索库规模，逗号分隔")
parser.add_argument("--robust_keep", type=str, default="0.5,0.7,0.9", help="下采样保留比例，逗号分隔")
parser.add_argument("--robust_sigma_deg", type=str, default="1e-5,2e-5,5e-5", help="噪声强度(度)，逗号分隔")