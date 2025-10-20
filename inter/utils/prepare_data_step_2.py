import os

import glob
import numpy as np


def process_point_clouds(point_path_prev, point_path_curr, to_save_path, tree_point=1024):
    octree_nodes_prev = np.load(point_path_prev)["octree_nodes"]
    octree_nodes_curr = np.load(point_path_curr)["octree_nodes"]

    length = min(len(octree_nodes_curr), len(octree_nodes_prev))

    for i in range(0, length, tree_point):
        filename = os.path.join(to_save_path, os.path.basename(point_path_curr).replace('.npz', f'_{i}.npz'))

        nodes_curr = octree_nodes_curr[i : i + tree_point]
        nodes_prev = octree_nodes_prev[i : i + tree_point]

        if len(nodes_curr) != tree_point:
            continue
        elif len(nodes_prev) != tree_point:
            continue
        else:
            np.savez_compressed(filename, octree_nodes_curr=nodes_curr, octree_nodes_prev=nodes_prev)

    return

if __name__ == '__main__':
    import glob
    import joblib

    for idx in range(0, 5):
        for seq in range(0, 12):
            point_paths = glob.glob(f"/zhu/data/NuScenes/train/process_rpcc_pccs/{idx:02d}/{seq:02d}/*")
            point_paths = sorted(point_paths)

            prev_point_paths = point_paths[:-1]
            curr_point_paths = point_paths[1:]

            print(len(prev_point_paths), len(curr_point_paths))
            assert len(prev_point_paths) == len(curr_point_paths)
            print(prev_point_paths[0], curr_point_paths[0])

            save_path = f"/zhu_disk1/data/NuScenes/train/process_rpcc_pccs_1024_dynamic/{idx:02d}/{seq:02d}"
            os.makedirs(save_path, exist_ok=True)

            joblib.Parallel(n_jobs=48, verbose=10)([
                joblib.delayed(process_point_clouds)(prev_point_paths[i], curr_point_paths[i], save_path)
                for i in range(len(prev_point_paths))
            ])
