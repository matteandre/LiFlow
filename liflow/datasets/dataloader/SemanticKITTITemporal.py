import torch
from torch.utils.data import Dataset
from liflow.utils.pcd_preprocess import load_poses
from liflow.utils.pcd_transforms import *
from liflow.utils.collations import point_set_to_sparse
from natsort import natsorted
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class TemporalKITTISet(Dataset):
    def __init__(self, data_dir, seqs, split, num_points, max_range):
        super().__init__()
        self.data_dir = data_dir

        self.n_clusters = 50
        self.num_points = num_points
        self.max_range = max_range

        self.split = split
        self.seqs = seqs
        self.cache_maps = {}

        self.datapath_list()

        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_list(self):
        self.points_datapath = []
        self.seq_poses = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq)
            point_seq_bin = natsorted(os.listdir(os.path.join(point_seq_path, 'velodyne')))
            poses = load_poses(os.path.join(point_seq_path, 'calib.txt'), os.path.join(point_seq_path, 'poses.txt'))
            p_full = np.load(f'{point_seq_path}/map_clean.npy') if self.split != 'test' else np.array([[1,0,0],[0,1,0],[0,0,1]])
            self.cache_maps[seq] = p_full
 
            for file_num in range(0, len(point_seq_bin)):
                self.points_datapath.append(os.path.join(point_seq_path, 'velodyne', point_seq_bin[file_num]))
                self.seq_poses.append(poses[file_num])

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return np.squeeze(points, axis=0)

    def __getitem__(self, index):
        seq_num = self.points_datapath[index].split('/')[-3]
        fname = self.points_datapath[index].split('/')[-1].split('.')[0]

        p_part = np.fromfile(self.points_datapath[index], dtype=np.float32)
        p_part = p_part.reshape((-1,4))[:,:3]
        
        if self.split != 'test':
            label_file = self.points_datapath[index].replace('velodyne', 'labels').replace('.bin', '.label')
            l_set = np.fromfile(label_file, dtype=np.uint32)
            l_set = l_set.reshape((-1))
            l_set = l_set & 0xFFFF
            static_idx = (l_set < 252) & (l_set > 1)
            p_part = p_part[static_idx]
            
        dist_part = np.sum(p_part**2, -1)**.5
        p_part = p_part[(dist_part < self.max_range) & (dist_part > 3.5)]
        p_part = p_part[p_part[:,2] > -4.]
        pose = self.seq_poses[index]

        p_map = self.cache_maps[seq_num]

        if self.split != 'test':
            trans = pose[:-1,-1]
            dist_full = np.sum((p_map - trans)**2, -1)**.5
            p_full = p_map[dist_full < self.max_range]
            p_full = np.concatenate((p_full, np.ones((len(p_full),1))), axis=-1)
            p_full = (p_full @ np.linalg.inv(pose).T)[:,:3]
            p_full = p_full[p_full[:,2] > -4.]
        else:
            p_full = p_part

        if self.split == 'train':
            p_concat = np.concatenate((p_full, p_part), axis=0)
            p_concat = self.transforms(p_concat)

            p_full = p_concat[:-len(p_part)]
            p_part = p_concat[-len(p_part):]

        n_part = int(self.num_points / 10.)

        return point_set_to_sparse(
            p_full,
            p_part,
            self.num_points,
            n_part,
            self.points_datapath[index],
        )

    def __len__(self):
        return self.nr_data
