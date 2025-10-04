import glob
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.nn.functional import softmax

from model.entropy_model import EntropyModel
from utils.torchac_utils import save_byte_stream

torch.set_float32_matmul_precision("high")

batch_size = 32
ctx_win = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_context(prev_ctx: torch.Tensor, curr_ctx: torch.Tensor):
    prev_ctx_len = prev_ctx.shape[0]
    curr_ctx_len = curr_ctx.shape[0]

    curr_pad_len = 1 if curr_ctx_len % 2 == 1 else 0

    if prev_ctx_len < curr_ctx_len + curr_pad_len:
        prev_pad_len = curr_ctx_len + curr_pad_len - prev_ctx_len
        prev_pad_ctx = torch.zeros((prev_pad_len, 4, 8), dtype=torch.int32)
        prev_ctx = torch.vstack((prev_ctx, prev_pad_ctx)).unsqueeze(1)
    else:
        prev_ctx = prev_ctx[:curr_ctx_len + curr_pad_len].unsqueeze(1)

    curr_pad = torch.zeros((curr_pad_len, 4, 8), dtype=torch.int32)
    curr_ctx = torch.vstack((curr_ctx, curr_pad)).unsqueeze(1)

    curr_pad_idx = torch.ones((curr_pad_len), dtype=torch.long) * -1
    curr_idx = torch.arange(curr_ctx_len, dtype=torch.long)
    curr_idx = torch.hstack((curr_idx, curr_pad_idx))
    curr_idx = curr_idx.reshape(-1, curr_ctx.shape[0]).transpose(0, 1)

    return prev_ctx, curr_ctx, curr_idx


def encode_level(model, prev_nodes, curr_nodes, level, max_level, elapsed):
    if level == 0:
        curr_ctx = torch.tensor([[
            [0, 0, 0, 2**max_level, 2**max_level, 2**max_level, 6, 0],
            [0, 0, 0, 2**max_level, 2**max_level, 2**max_level, 6, 0],
            [0, 0, 0, 2**max_level, 2**max_level, 2**max_level, 6, 0],
            [0, 0, 0, 2**max_level, 2**max_level, 2**max_level, 6, 0],
        ]], dtype=torch.int32)

        length = 1
        prob = torch.zeros((length + 1, 255))

        prev_pad_ctx, curr_pad_ctx, pad_idx = process_context(prev_nodes, curr_ctx)

        output, elapsed = model.encode(prev_pad_ctx.to(device), curr_pad_ctx.to(device), level, elapsed)
        prob[pad_idx] = softmax(output, dim=-1)

        occupys = curr_nodes[:, -1, 0]
        bits = save_byte_stream(prob[:-1], occupys, f"output/{level}.bin")
    else:
        length = curr_nodes.shape[0]

        if length % 2 == 1:
            prob = torch.zeros((length + 1, 255))
        else:
            prob = torch.zeros((length, 255))

        prev_pad_ctx, curr_pad_ctx, pad_idx = process_context(prev_nodes, curr_nodes)

        pad_ctx_len = curr_pad_ctx.shape[0]

        win_q = pad_ctx_len // ctx_win
        win_r = pad_ctx_len % ctx_win

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

                    output, elapsed = model.encode(prev_ctx.to(device), curr_ctx.to(device), level, elapsed)
                    prob[idx] = softmax(output, dim=-1)

            if batch_r > 0:
                idx = win_idx[-batch_r:].transpose(0, 1)
                prev_ctx = prev_win_ctx[-batch_r:].transpose(0, 1)
                curr_ctx = curr_win_ctx[-batch_r:].transpose(0, 1)

                output, elapsed = model.encode(prev_ctx.to(device), curr_ctx.to(device), level, elapsed)
                prob[idx] = softmax(output, dim=-1)

        if win_r > 0:
            idx = pad_idx[-win_r:]
            prev_ctx = prev_pad_ctx[-win_r:]
            curr_ctx = curr_pad_ctx[-win_r:]

            output, elapsed = model.encode(prev_ctx.to(device), curr_ctx.to(device), level, elapsed)
            prob[idx] = softmax(output, dim=-1)

        prob = prob[:-1] if length % 2 == 1 else prob
        occupys = curr_nodes[:, -1, 0]
        bits_1 = save_byte_stream(prob[0::2], occupys[0::2], f"output/{level}_1.bin")
        bits_2 = save_byte_stream(prob[1::2], occupys[1::2], f"output/{level}_2.bin")
        bits = bits_1 + bits_2

    return bits, elapsed

@torch.no_grad()
def encode(model, prev_nodes, curr_nodes, max_level):
    prev_nodes = torch.tensor(prev_nodes, dtype=torch.int32)
    curr_nodes = torch.tensor(curr_nodes, dtype=torch.int32)

    prev_nodes[:, :, 0] = prev_nodes[:, :, 0] - 1
    curr_nodes[:, :, 0] = curr_nodes[:, :, 0] - 1

    prev_levels = prev_nodes[:, -1, 1]
    curr_levels = curr_nodes[:, -1, 1]

    bits = 0
    elapsed = 0

    for level in range(0, max_level):
        level_bits, elapsed = encode_level(model, prev_nodes[prev_levels==level], curr_nodes[curr_levels==level], level, max_level, elapsed)
        bits += level_bits

    return bits, elapsed


def main(model, prev_path, curr_path):
    prev_frame = np.load(prev_path)
    curr_frame = np.load(curr_path)

    pt_num    = curr_frame["pt_num"]
    max_level = curr_frame["level"]

    bits, elapsed = encode(model, prev_frame["octree_nodes"], curr_frame["octree_nodes"], max_level)

    return bits / pt_num, bits, pt_num, elapsed


if __name__ == "__main__":
    checkpoint_path = "ckpt/dynamic_kitti.ckpt"
    model = EntropyModel(checkpoint_path).eval().to(device)

    logdir = f"result_dynamic_kitti.csv"

    sequences = ["11", "12", "13"]

    level_list = [11, 12, 13, 14, 15, 16]

    for seq in sequences:
        print(seq)
        list_orifile = glob.glob(f"./data/SemanticKITTI/test/dataset/sequences/{seq}/**/*.bin")
        list_orifile.sort()

        prev_list = list_orifile[:-1]
        curr_list = list_orifile[1:]

        assert len(prev_list) == len(curr_list)
        # print(prev_list[-1], curr_list[-1])

        for idx, path in enumerate(tqdm(curr_list)):
            new_row = {"filedir": path}

            for level in level_list:
                basename = int(path.split("/")[-1].split(".")[0])
                curr_name = f"./data/test/{seq}/{basename    :06d}_{level}.npz"
                prev_name = f"./data/test/{seq}/{basename - 1:06d}_{level}.npz"
                # print(prev_name, curr_name)

                bpp, bits, points, elapsed = main(model, prev_name, curr_name)

                new_row[f"r{level}_bpp"] = bpp
                new_row[f"r{level}_bits"] = bits
                new_row[f"r{level}_points"] = points

            results = pd.DataFrame([new_row])

            if not os.path.exists(logdir):
                results.to_csv(logdir, index=False)
            else:
                results.to_csv(logdir, mode="a", header=False, index=False)
