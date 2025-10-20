import numpy as np
import torch
import csv
import yaml
from easydict import EasyDict

class TreeNode:
    my_shift = np.array(
        (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        )
    )

    def __init__(self, min_bound, node_idx, curr_occu, level, par_ctx, voxel_size_by_level):
        self.min_bound = min_bound
        self.node_idx = node_idx
        self.curr_occu = curr_occu
        self.child_node_ls = []
        self.level = level
        self.par_ctx = par_ctx
        self.voxel_size_by_level = voxel_size_by_level
        self.init_origin_coords()

    def init_origin_coords(self):
        voxel_size = self.voxel_size_by_level[self.level]
        self.origin = self.min_bound + self.my_shift[self.node_idx] * voxel_size
        self.coords = self.origin + voxel_size * 0.5

    def get_children_nodes(self, occu_symbols):
        assert self.curr_occu == None
        self.curr_occu = int(occu_symbols)

        min_bound = self.origin
        level = self.level + 1

        occupy = '{0:08b}'.format(self.curr_occu)
        idx_ls = [i for i, e in enumerate(occupy) if e != "0"]

        for _, node_idx in enumerate(idx_ls):
            curr_occu = None
            par_ctx = self.get_decode_context()[1:]
            child_node = TreeNode(min_bound, node_idx, curr_occu, level, par_ctx, self.voxel_size_by_level)
            self.child_node_ls.append(child_node)

        return self.child_node_ls

    def get_decode_context(self):
        # get the context of node
        if self.par_ctx is None:
            occupy = 0 if self.curr_occu is None else self.curr_occu
            curr = torch.tensor([occupy, self.level, self.node_idx, *self.coords])
            context = torch.repeat_interleave(curr.unsqueeze(0), 4, dim=0)
        else:
            prev = self.par_ctx

            # occupy = prev[-1, 0] if self.curr_occu is None else self.curr_occu
            occupy = 0 if self.curr_occu is None else self.curr_occu
            curr = torch.tensor([occupy, self.level, self.node_idx, *self.coords])

            context = torch.cat((prev, curr.unsqueeze(0)), dim=0)

        return context


def get_voxel_size_by_level_dict(max_bound, min_bound, level):
    voxel_size_by_level = dict()

    for i in range(level + 1):
        voxel_size_by_level.update({i: (max_bound - min_bound) / (2 ** i)})

    return voxel_size_by_level


class PCTransformer:
    def __init__(self, lidar_cfg=None, channel_distribute_csv=None):
        if channel_distribute_csv is not None:
            self.even_dist = False
            channel = []
            vertical_angle = []

            with open(channel_distribute_csv, "r") as fin:
                reader = csv.DictReader(fin)
                for r in reader:
                    channel.append(int(r["channel"]))
                    vertical_angle.append(float(r["vertical_angle"]))

            self.vertical_angle = np.radians(np.array(vertical_angle))
        else:
            self.even_dist = True

        with open(lidar_cfg, "r") as f:
            try:
                config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                config = yaml.load(f)

        lidar_config = EasyDict(config)

        self.vertical_max = lidar_config.VERTICAL_ANGLE_MAX * (np.pi / 180)
        self.vertical_min = lidar_config.VERTICAL_ANGLE_MIN * (np.pi / 180)

        self.vertical_FOV = self.vertical_max - self.vertical_min
        self.horizontal_FOV = lidar_config.HORIZONTAL_FOV * (np.pi / 180)

        self.H = lidar_config.RANGE_IMAGE_HEIGHT
        self.W = lidar_config.RANGE_IMAGE_WIDTH

    def calculate_vertical_angle(self, points):
        return np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], 2, -1))

    def calculate_horizon_angle(self, points):
        return np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)

    def point_cloud_to_range_image(self, points, level):
        if len(points.shape) == 1:
            points = points[np.newaxis, :]

        qs = 400 / (2 ** level - 1)
        offset = np.array([-2**(level-1), -2**(level-1), -2**(level-1)], dtype=np.int32)
        points = (points + offset) * qs

        # horizontal index h
        horizontal_angle = self.calculate_horizon_angle(points)
        col = np.rint(horizontal_angle / self.horizontal_FOV * self.W)
        col = col % self.W

        # vertical index w
        vertical_angle = self.calculate_vertical_angle(points)
        if self.even_dist:
            vertical_resolution = (self.vertical_max - self.vertical_min) / (self.H - 1)
            row = np.rint((vertical_angle - self.vertical_min) / vertical_resolution)
        else:
            vertical_angle_dif = np.expand_dims(self.vertical_angle, 0) - np.expand_dims(vertical_angle, 1)
            row = np.argmin(np.abs(vertical_angle_dif), -1)

        row[np.where(row >= self.H)] = self.H - 1
        row[np.where(row < 0)] = 0

        return row.astype(np.int32), col.astype(np.int32)
