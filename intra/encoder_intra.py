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


def process_context(ctx):
    ctx_len = ctx.shape[0]

    pad_len = 1 if ctx_len % 2 == 1 else 0

    pad_ctx = torch.zeros((pad_len, 4, 8), dtype=torch.int32)
    ctx = torch.vstack((ctx, pad_ctx)).unsqueeze(1)

    pad_idx = torch.ones((pad_len), dtype=torch.long) * -1
    idx = torch.arange(ctx_len)
    idx = torch.hstack((idx, pad_idx))
    idx = idx.reshape(-1, ctx.shape[0]).transpose(0, 1)

    return ctx, idx


def encode_level(model, octree_nodes, level, max_level, elapsed):
    if level == 0:
        ctx = torch.tensor([[
            [0, 0, 0, 2**max_level, 2**max_level, 2**max_level, 6, 0],
            [0, 0, 0, 2**max_level, 2**max_level, 2**max_level, 6, 0],
            [0, 0, 0, 2**max_level, 2**max_level, 2**max_level, 6, 0],
            [0, 0, 0, 2**max_level, 2**max_level, 2**max_level, 6, 0],
        ]], dtype=torch.int32)

        length = 1
        prob = torch.zeros((length + 1, 255))

        pad_ctx, pad_idx = process_context(ctx)

        output, elapsed = model.encode(pad_ctx.to(device), level, elapsed)
        prob[pad_idx] = softmax(output, dim=-1)

        occupys = octree_nodes[:, -1, 0]
        bits = save_byte_stream(prob[:-1], occupys, f"output/{level}.bin")
    else:
        length = octree_nodes.shape[0]

        if length % 2 == 1:
            prob = torch.zeros((length + 1, 255))
        else:
            prob = torch.zeros((length, 255))

        pad_ctx, pad_idx = process_context(octree_nodes)
        pad_ctx_len = pad_ctx.shape[0]

        win_q = pad_ctx_len // ctx_win
        win_r = pad_ctx_len % ctx_win

        if win_q > 0:
            win_idx = pad_idx[:win_q * ctx_win].reshape(-1, ctx_win)
            win_ctx = pad_ctx[:win_q * ctx_win].reshape(-1, ctx_win, 4, 8)

            batch_q = win_q // batch_size
            batch_r = win_q % batch_size

            if batch_q > 0:
                for i in range(0, batch_q):
                    idx = win_idx[i * batch_size : (i + 1) * batch_size].transpose(0, 1)
                    ctx = win_ctx[i * batch_size : (i + 1) * batch_size].transpose(0, 1)
                    output, elapsed = model.encode(ctx.to(device), level, elapsed)
                    prob[idx] = softmax(output, dim=-1)

            if batch_r > 0:
                idx = win_idx[-batch_r:].transpose(0, 1)
                ctx = win_ctx[-batch_r:].transpose(0, 1)

                output, elapsed = model.encode(ctx.to(device), level, elapsed)
                prob[idx] = softmax(output, dim=-1)

        if win_r > 0:
            idx = pad_idx[-win_r:]
            ctx = pad_ctx[-win_r:]

            output, elapsed = model.encode(ctx.to(device), level, elapsed)
            prob[idx] = softmax(output, dim=-1)

        prob = prob[:-1] if length % 2 == 1 else prob
        occupys = octree_nodes[:, -1, 0]
        bits_1 = save_byte_stream(prob[0::2], occupys[0::2], f"output/{level}_1.bin")
        bits_2 = save_byte_stream(prob[1::2], occupys[1::2], f"output/{level}_2.bin")
        bits = bits_1 + bits_2

    return bits, elapsed

@torch.no_grad()
def encode(model, octree_nodes, max_level):
    octree_nodes = torch.tensor(octree_nodes, dtype=torch.int32)
    octree_nodes[:, :, 0] = octree_nodes[:, :, 0] - 1

    levels = octree_nodes[:, -1, 1]

    bits = 0
    elapsed = 0

    for level in range(0, max_level):
        level_bits, elapsed = encode_level(model, octree_nodes[levels==level], level, max_level, elapsed)
        bits += level_bits

    return bits, elapsed


def main(model, file_name):
    frame_data = np.load(file_name)

    pt_num = frame_data["pt_num"]
    max_level = frame_data["level"]

    bits, elapsed = encode(model, frame_data["octree_nodes"], max_level)

    print("Total elapsed time: ", elapsed)

    return bits / pt_num, bits, pt_num


if __name__ == "__main__":
    checkpoint_path = "ckpt/static_nuscenes.ckpt"
    model = EntropyModel(checkpoint_path).eval().to(device)
    logdir = "result_static.csv"

    sequences = ["05", "06", "07", "08", "09"]
    level_list = [11, 12, 13, 14, 15, 16]

    FIRST = True

    for seq in sequences:
        print(seq)
        list_orifile = glob.glob(f"/zhu/data/nuscenes/test/{seq}/**/*.bin", recursive=True)
        list_orifile.sort()

        print(len(list_orifile))

        for idx, path in enumerate(tqdm(list_orifile)):
            new_row = {"filedir": path}

            for level in level_list:
                basename = int(path.split("/")[-1].split(".")[0])
                curr_name = f"./test/{seq}/{basename:03d}_{level}.npz"

                bpp, bits, points = main(model, curr_name)

                new_row[f"r{level}_bpp"] = bpp
                new_row[f"r{level}_bits"] = bits
                new_row[f"r{level}_points"] = points

            results = pd.DataFrame([new_row])

            if FIRST:
                results.to_csv(logdir, index=False)
                FIRST = False
            else:
                results.to_csv(logdir, mode="a", header=False, index=False)
