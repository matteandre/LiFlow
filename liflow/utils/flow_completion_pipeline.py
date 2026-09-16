import numpy as np
import torch
import open3d as o3d
from pytorch_lightning import LightningModule
import yaml
import os
import tqdm
from natsort import natsorted
import click
import time
from liflow.utils.helpers import load_model_from_config

class FlowCompletion(LightningModule):
    def __init__(self, flow_path , refine_path, steps=10):
        super().__init__()

        ckpt_cfg = yaml.safe_load(open(flow_path.split('checkpoints')[0] + '/hparams.yaml'))
        self.ckpt_cfg = ckpt_cfg
        self.model = load_model_from_config(ckpt_cfg['model'], flow_path).cuda()
        self.model.save_hyperparameters(ckpt_cfg)


        ckpt_cfg_refine = yaml.safe_load(open(refine_path.split('checkpoints')[0] + '/hparams.yaml'))
        
        self.model_refine = load_model_from_config(ckpt_cfg_refine['model'], refine_path).cuda()
        self.model_refine.save_hyperparameters(ckpt_cfg_refine)
        self.model.eval()
        self.model_refine.eval()
        self.cuda()
        self.steps = steps

        
        exp_dir = flow_path.split('/')[-1].split('.')[0].replace('=','')
        os.makedirs(f'./results/{exp_dir}', exist_ok=True)
        with open(f'./results/{exp_dir}/exp_config.yaml', 'w+') as exp_config:
            yaml.dump(self.hparams, exp_config)
                                                                              
    def preprocess_scan(self, scan):
        scan = scan[:,:3]
        dist = np.sqrt(np.sum((scan)**2, -1))
        scan = scan[(dist < self.ckpt_cfg['data']['params']['max_range']) & (dist > 3.5)][:,:3]
        # use farthest point sampling
        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan)
        pcd_scan = pcd_scan.farthest_point_down_sample(int(self.ckpt_cfg['data']['params']['num_points'] / 10))
        scan = torch.tensor(np.array(pcd_scan.points)).float().cuda()
        
        cond = torch.tensor(np.array(pcd_scan.points)).float().cuda()[None,:,:]
        
        scan = scan.repeat(10,1)
        scan = scan[None,:,:]

        return scan, cond

    def postprocess_scan(self, completed_scan, input_scan):
        dist = np.sqrt(np.sum((completed_scan)**2, -1))
        post_scan = completed_scan[dist < self.ckpt_cfg['data']['params']['max_range']]
        max_z = input_scan[...,2].max().item()
        min_z = (input_scan[...,2].mean() - 2 * input_scan[...,2].std()).item()
        post_scan = post_scan[(post_scan[...,2] < max_z) & (post_scan[...,2] > min_z)]
        return post_scan

    def complete_scan(self, scan):
        scan, cond = self.preprocess_scan(scan)
        x_0 = scan + torch.randn(scan.shape, device=self.device)

        x_cond = self.model.points_to_tensor(cond)
        x_uncond = self.model.points_to_tensor(torch.zeros_like(cond))

        completed_scan = self.model.generate(x_0, x_cond, x_uncond, sample_kwargs={'num_steps':10}).cpu().detach().numpy()
        post_scan = self.postprocess_scan(completed_scan, scan)


        refine_in = self.model_refine.points_to_tensor(post_scan[None,:,:])
        offset = self.model_refine.forward_refine(refine_in).reshape(-1,6,3).cpu().detach().numpy()

        refine_complete_scan = post_scan[:,None,:] + offset
        return post_scan, refine_complete_scan.reshape(-1,3)

def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,4))[:,:3]
    elif pcd_file.endswith('.ply'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points).astype(np.float32)
    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")

@click.command()
@click.option('--path', '-p', type=str, default='Datasets/test', help='path to the dataset')
@click.option('--flow', '-f', type=str, default='checkpoints/flow_net.ckpt', help='path to the checkpoint for LiFlow model')
@click.option('--refine', '-r', type=str, default='checkpoints/refine_net.ckpt', help='path to the checkpoint for refinement net')
@click.option('--steps', '-s', type=int, default=10, help='number of Euler method steps')

def main(path, flow, refine, steps):
    exp_dir = flow.split('/')[-1].split('.')[0].replace('=','')

    flow_completion = FlowCompletion(
            flow, refine, steps
        )

    os.makedirs(f'./results/{exp_dir}/flow', exist_ok=True)
    os.makedirs(f'./results/{exp_dir}/refine', exist_ok=True)

    path = os.path.join(path, 'velodyne')

    for pcd_path in tqdm.tqdm(natsorted(os.listdir(path))):
        pcd_file = os.path.join(path, pcd_path)
        points = load_pcd(pcd_file)


        dist = np.sqrt(np.sum((points)**2, -1))
        scan = points[(dist < 50) & (dist > 3.5)][:,:3]
    
        start = time.time()
        flow_scan, refine_scan = flow_completion.complete_scan(scan)
        end = time.time()
        print(f'took: {end - start}s')
       

        pcd_flow = o3d.geometry.PointCloud()
        pcd_flow.points = o3d.utility.Vector3dVector(flow_scan)
        pcd_flow.estimate_normals()
        o3d.io.write_point_cloud(f'./results/{exp_dir}/flow/{pcd_path.split(".")[0]}.ply', pcd_flow)
        
        
        pcd_refine = o3d.geometry.PointCloud()
        pcd_refine.points = o3d.utility.Vector3dVector(refine_scan)
        pcd_refine.estimate_normals()
        o3d.io.write_point_cloud(f'./results/{exp_dir}/refine/{pcd_path.split(".")[0]}.ply', pcd_refine)

if __name__ == '__main__':
    main()
