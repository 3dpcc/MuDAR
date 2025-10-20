import os

import numpy as np
import open3d as o3d


def read_ply_ascii(filedir, dtype="float32"):
    files = open(filedir, "r")
    data = []

    for _, line in enumerate(files):
        wordslist = line.split(" ")
        try:
            line_values = []
            for _, v in enumerate(wordslist):
                if v == "\n":
                    continue
                line_values.append(float(v))
        except ValueError:
            continue
        data.append(line_values)

    data = np.array(data)
    coords = data[:, 0:3].astype(dtype)

    return coords


def read_ply_o3d(filedir, dtype="float32"):
    pcd = o3d.io.read_point_cloud(filedir)
    coords = np.asarray(pcd.points, dtype)

    return coords


def read_bin(filedir, dtype="float32"):
    data = np.fromfile(filedir, dtype).reshape(-1, 4)
    coords = data[:, :3]

    return coords

def read_bin_nuscences(filedir, dtype=np.float32):
    data = np.fromfile(filedir, dtype).reshape(-1, 5)
    coords = data[:, :3]

    return coords


def read_coords(filedir, dtype="float32"):
    file_extension = filedir.split(".")[-1]
    match file_extension:
        case "ply":
            coords = read_ply_ascii(filedir, dtype)
            # coords = read_ply_o3d(filedir, dtype)
        case "bin":
            coords = read_bin(filedir, dtype)
        case _:
            raise ValueError("Unknown file extension.")

    return coords


def write_ply_ascii(filedir, coords, dtype="float32"):
    if os.path.exists(filedir):
        os.remove(filedir)

    with open(filedir, "a+") as f:
        f.writelines(["ply\n", "format ascii 1.0\n"])
        f.write(f"element vertex {coords.shape[0]}\n")
        f.writelines(["property float x\n", "property float y\n", "property float z\n"])
        f.write("end_header\n")

        coords = coords.astype(dtype)
        for p in coords:
            f.writelines(f"{p[0]} {p[1]} {p[2]}\n")
    return
