from collections import deque

import numpy as np
import torch
import yaml

from model.entropy_model import EntropyModel
from utils.octree import PCTransformer, TreeNode, get_voxel_size_by_level_dict
from utils.torchac_utils import get_symbol_from_byte_stream
from torch.nn.functional import softmax

batch_size = 32
ctx_win = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = yaml.safe_load(open("config.yaml", "r"))

lidar_cfg = "config/Velodyne_HDL_64E.yaml"
pc_transformer = PCTransformer(lidar_cfg)

torch.set_float32_matmul_precision("high")


def process_context(prev_ctx, curr_ctx, level):
    prev_ctx = torch.tensor(prev_ctx, dtype=torch.int32)
    curr_ctx = torch.stack(curr_ctx, dim=0).to(dtype=torch.int32)

    laser_0, phi_0 = pc_transformer.point_cloud_to_range_image(curr_ctx[:, 0, 3:6].cpu().numpy(), level)
    laser_1, phi_1 = pc_transformer.point_cloud_to_range_image(curr_ctx[:, 1, 3:6].cpu().numpy(), level)
    laser_2, phi_2 = pc_transformer.point_cloud_to_range_image(curr_ctx[:, 2, 3:6].cpu().numpy(), level)
    laser_3, phi_3 = pc_transformer.point_cloud_to_range_image(curr_ctx[:, 3, 3:6].cpu().numpy(), level)

    laser = torch.tensor(np.concatenate((laser_0[:, np.newaxis, np.newaxis], laser_1[:, np.newaxis, np.newaxis], laser_2[:, np.newaxis, np.newaxis], laser_3[:, np.newaxis, np.newaxis]), axis=1))
    phi   = torch.tensor(np.concatenate((phi_0[:, np.newaxis, np.newaxis], phi_1[:, np.newaxis, np.newaxis], phi_2[:, np.newaxis, np.newaxis], phi_3[:, np.newaxis, np.newaxis]), axis=1))

    curr_ctx = torch.cat((curr_ctx, laser, phi), dim=-1)

    prev_ctx_len = prev_ctx.shape[0]
    curr_ctx_len = curr_ctx.shape[0]

    curr_pad_len = 1 if curr_ctx_len % 2 == 1 else 0

    if prev_ctx_len < curr_ctx_len + curr_pad_len:
        prev_pad_len = curr_ctx_len + curr_pad_len - prev_ctx_len
        prev_pad_ctx = torch.zeros((prev_pad_len, 4, 8), dtype=torch.int32)
        prev_ctx = torch.vstack((prev_ctx, prev_pad_ctx)).unsqueeze(1)
    else:
        prev_ctx = prev_ctx[:curr_ctx_len + curr_pad_len].unsqueeze(1)

    curr_pad_ctx = torch.zeros((curr_pad_len, 4, 8), dtype=torch.int32)
    curr_ctx = torch.vstack((curr_ctx, curr_pad_ctx)).unsqueeze(1)

    curr_pad_idx = torch.ones((curr_pad_len), dtype=torch.long) * -1
    curr_idx = torch.arange(curr_ctx_len, dtype=torch.long)
    curr_idx = torch.hstack((curr_idx, curr_pad_idx))
    curr_idx = curr_idx.reshape(-1, curr_ctx.shape[0]).transpose(0, 1)

    return prev_ctx, curr_ctx, curr_idx


def decode_level(model, prev_nodes, nodes, level, max_level, elapsed):
    if level == 0:
        with open("output/0.bin", "rb") as fin:
            byte_stream = fin.read()

        curr_length = len(nodes)

        ori_ctx = [node.get_decode_context() for node in nodes]
        prev_pad_ctx, curr_pad_ctx, pad_idx = process_context(prev_nodes, ori_ctx, max_level)

        prob = torch.zeros((curr_length + 1, 255))
        output, elapsed = model.decode(prev_pad_ctx.to(device), curr_pad_ctx.to(device), elapsed)
        prob[pad_idx] = softmax(output, dim=-1)

        syms = get_symbol_from_byte_stream(byte_stream, prob[:-1])
    else:
        with open(f"output/{level}_1.bin", "rb") as fin:
            byte_stream_1 = fin.read()

        with open(f"output/{level}_2.bin", "rb") as fin:
            byte_stream_2 = fin.read()

        prev_length = prev_nodes.shape[0]
        curr_length = len(nodes)

        ori_ctx = [node.get_decode_context() for node in nodes]
        prev_pad_ctx, curr_pad_ctx, pad_idx = process_context(prev_nodes, ori_ctx, max_level)

        prev_pad_ctx[:prev_length, :, :, 0] = prev_pad_ctx[:prev_length, :, :, 0] - 1
        curr_pad_ctx[:curr_length, :, :, 0] = curr_pad_ctx[:curr_length, :, :, 0] - 1
        curr_pad_ctx[:, :, -1, 0] = 255

        # Group 1
        syms_1, elapsed = decode_group(model, curr_length, prev_pad_ctx, curr_pad_ctx, pad_idx, 1, byte_stream_1, elapsed)

        # Group 2
        curr_pad_ctx[0::2, :, -1, 0] = syms_1.reshape(-1, 1)
        syms_2, elapsed = decode_group(model, curr_length, prev_pad_ctx, curr_pad_ctx, pad_idx, 2, byte_stream_2, elapsed)

        syms = np.zeros((curr_length), dtype=np.int32)
        syms[0::2] = syms_1
        syms[1::2] = syms_2

    return syms + 1, elapsed


def decode_group(model, length, prev_pad_ctx, curr_pad_ctx, pad_idx, group, byte_stream, elapsed):
    if length % 2 == 1:
        prob = torch.zeros((length + 1, 255))
    else:
        prob = torch.zeros((length, 255))

    padding_ctx_len = curr_pad_ctx.shape[0]

    win_q = padding_ctx_len // ctx_win
    win_r = padding_ctx_len % ctx_win

    if win_q > 0:
        win_idx = pad_idx[:win_q * ctx_win].reshape(-1, ctx_win)
        prev_win_ctx = prev_pad_ctx[:win_q * ctx_win].reshape(-1, ctx_win, 4, 8)
        curr_win_ctx = curr_pad_ctx[:win_q * ctx_win].reshape(-1, ctx_win, 4, 8)

        batch_q = win_q // batch_size
        batch_r = win_q % batch_size

        if batch_q > 0:
            for i in range(0, batch_q):
                idx = win_idx[i * batch_size : (i + 1) * batch_size].transpose(0, 1)
                prev_ctx = prev_win_ctx[i * batch_size : (i + 1) * batch_size].transpose(0, 1)
                curr_ctx = curr_win_ctx[i * batch_size : (i + 1) * batch_size].transpose(0, 1)

                output, elapsed = model.decode(prev_ctx.to(device), curr_ctx.to(device), elapsed)
                prob[idx] = softmax(output, dim=-1)

        if batch_r > 0:
            idx = win_idx[-batch_r:].transpose(0, 1)
            prev_ctx = prev_win_ctx[-batch_r:].transpose(0, 1)
            curr_ctx = curr_win_ctx[-batch_r:].transpose(0, 1)

            output, elapsed = model.decode(prev_ctx.to(device), curr_ctx.to(device), elapsed)
            prob[idx] = softmax(output, dim=-1)

    if win_r > 0:
        idx = pad_idx[-win_r:]
        prev_ctx = prev_pad_ctx[-win_r:]
        curr_ctx = curr_pad_ctx[-win_r:]

        output, elapsed = model.decode(prev_ctx.to(device), curr_ctx.to(device), elapsed)
        prob[idx] = softmax(output, dim=-1)

    prob = prob[:-1] if length % 2 == 1 else prob
    prob = prob[0::2] if group == 1 else prob[1::2]

    syms = get_symbol_from_byte_stream(byte_stream, prob)

    return syms, elapsed


@torch.no_grad
def decode(model, prev_frame, curr_frame):
    prev_nodes = prev_frame["octree_nodes"].astype(np.int32)
    curr_nodes = curr_frame["octree_nodes"].astype(np.int32)

    occupys = curr_nodes[:, -1, 0]

    max_level = curr_frame["level"]
    max_bound = curr_frame["max_bound"]
    min_bound = curr_frame["min_bound"]

    prev_levels = prev_nodes[:, -1, 1]
    curr_levels = curr_nodes[:, -1, 1]

    # Dictionary to obtain voxel size by level
    voxel_size_by_level = get_voxel_size_by_level_dict(max_bound, min_bound, max_level)

    # Begin decoding
    elapsed = 0

    total_symbol = []

    nodes = deque()
    nodes.append(TreeNode(min_bound, node_idx=0, curr_occu=None, level=0, par_ctx=None, voxel_size_by_level=voxel_size_by_level))

    for level in range(max_level):
        length = len(nodes)

        syms, elapsed = decode_level(model, prev_nodes[prev_levels==level], nodes, level, max_level, elapsed)

        total_symbol.extend(syms)
        assert total_symbol == occupys[: len(total_symbol)].tolist()

        new_nodes = deque()
        for i in range(length):
            nodes[i].get_children_nodes(syms[i])
            new_nodes.extend(nodes[i].child_node_ls)

        nodes = new_nodes

    return nodes, elapsed


if __name__ == "__main__":
    prev_path = "xxxxx"
    curr_path = "xxxxx"
    model_path = "./ckpt/dynamic_ford_lossy.ckpt"

    prev_frame = np.load(prev_path)
    curr_frame = np.load(curr_path)
    model = EntropyModel(model_path).eval().to(device)

    nodes, elapsed = decode(model, prev_frame, curr_frame)

    print("Decoding time: ", elapsed)
