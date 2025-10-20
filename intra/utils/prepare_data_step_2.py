import os

import glob
import numpy as np

def process_point_cloud(point_path, save_path, tree_point=1024):
    octree_nodes = np.load(point_path)["octree_nodes"]

    length = len(octree_nodes)

    chunks = length // tree_point

    for i in range(chunks):
        file_name = os.path.join(save_path, os.path.basename(point_path).replace('.npz', f'_{i}.npz'))
        nodes = octree_nodes[tree_point * i : tree_point * (i + 1)]

        np.savez_compressed(file_name, octree_nodes=nodes)

    return


if __name__ == '__main__':
    import glob
    import joblib

    for idx in range(0, 5):
        for seq in range(0, 12):
            point_paths = glob.glob(f"/zhu/data/NuScenes/train/process_rpcc_pccs/{idx:02d}/{seq:02d}/*")
            point_paths = sorted(point_paths)
            print(len(point_paths))

            save_path = f"/zhu_disk1/data/NuScenes/train/process_rpcc_pccs_1024_static/{idx:02d}/{seq:02d}"
            os.makedirs(save_path, exist_ok=True)

            joblib.Parallel(n_jobs=48, verbose=10)([
                joblib.delayed(process_point_cloud)(path, save_path)
                for path in point_paths
            ])
