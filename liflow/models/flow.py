import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
from flow_matching.path import CondOTProbPath
from flow_matching.solver.ode_solver import ODESolver
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
from liflow.utils.helpers import instantiate_from_config
from liflow.utils.collations import *
from liflow.utils.metrics import ChamferDistance, PrecisionRecall
import os

_ATOL = 1e-6
_RTOL = 1e-3

class FlowModel(LightningModule):
    def __init__(
            self,
            model_cfg,
            partial_enc_cfg,
            sigma_min = 0.0,
            uncond_prob = 0.1,
            uncond_w = 6,
            resolution=0.05,
            lr = 0.0001
        ):
       
        super().__init__()
        self.partial_enc = instantiate_from_config(partial_enc_cfg) 
        self.model = instantiate_from_config(model_cfg)
        self.uncond_prob = uncond_prob
        self.uncond_w = uncond_w
        self.resolution = resolution
        self.sigma_min = sigma_min
        self.lr = lr
        self.path = CondOTProbPath()
        self.solver = ODESolver(velocity_model=self.ode_fn)
        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.resolution ,2*self.resolution,100)
    

    def forward(self, x_full, x_full_sparse, x_part, t):
        part_feat = self.partial_enc(x_part)
        out = self.model(x_full, x_full_sparse, part_feat, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)
    
    def feats_to_coord(self, p_feats, bs):
        p_feats = p_feats.reshape(bs,-1,3)
        p_coord = torch.round(p_feats / self.resolution)
        return p_coord.reshape(-1,3)
    
    def points_to_tensor(self, x):
        x_feats = ME.utils.batched_coordinates(list(x[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord[:,1:] = self.feats_to_coord(x_feats[:,1:], x.shape[0])
        x_tf = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()
        return x_tf
    
    def compute_x_u_t(self, x_0, x_1, t):
        _,_,match_x_01 = knn_points(x_0, x_1, return_nn=True)
        match_x_01 = match_x_01.reshape(x_0.shape)
        x_t = t[:,None,None]*match_x_01 + (1-t)[:,None,None]*x_0
        u_t = match_x_01 - x_0
        return x_t, u_t

    def training_step(self, batch:dict, batch_idx):
        torch.cuda.empty_cache()
        x_1 = batch['pcd_full']
        x_0 = batch['pcd_part'].repeat(1,10,1)
        noise = torch.randn(x_0.shape, device=self.device)
        x_0 = x_0 + noise
        t = torch.rand(x_0.shape[0], device=self.device)
        x_t, u_t = self.compute_x_u_t(x_0, x_1, t)

        if self.sigma_min > 0:
            x_t += self.sigma_min * torch.randn_like(x_t)

        x_t = self.points_to_tensor(x_t)
        # for classifier-free guidance switch between conditional and unconditional training
        if torch.rand(1) > self.uncond_prob or batch['pcd_full'].shape[0] == 1:
            x_part = self.points_to_tensor(batch['pcd_part'])
        else:
            x_part = self.points_to_tensor(torch.zeros_like(batch['pcd_part']))

        v_t = self.forward(x_t, x_t.sparse(), x_part, t)

        loss_mse = F.mse_loss(v_t, u_t)
        loss_cfd, _ = chamfer_distance(x_0+v_t, x_1)
        loss = loss_mse + 0.1*loss_cfd

        self.log('train/loss_mse', loss_mse, prog_bar=True)
        self.log('train/loss_cfd', loss_cfd, prog_bar=True)
        self.log('train/loss', loss)
        torch.cuda.empty_cache()
        return loss
    
    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)
        return x_uncond + self.uncond_w * (x_cond - x_uncond)
    
    def generate(self, x_0, x_cond, x_uncond, sample_kwargs={}):
       
        num_steps = sample_kwargs.get("num_steps", 10)
        time_grid = torch.linspace(0, 1, num_steps).to(self.device)
        results = self.solver.sample(
            time_grid=time_grid,
            x_init=x_0,
            method=sample_kwargs.get("method", "euler"),
            return_intermediates=False,
            atol=sample_kwargs.get("atol", _ATOL),
            rtol=sample_kwargs.get("rtol", _RTOL),
            step_size=None,
            x_cond=x_cond,
            x_uncond=x_uncond,
        )

        return results
    
    def ode_fn(self, x, t, x_cond, x_uncond):
        torch.cuda.empty_cache()

        if t.numel() == 1:
            t = t.expand(x.size(0))

        bs= x.shape[0]
        x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, bs)
        x_t = self.points_to_tensor(x)
        return self.classfree_forward(x_t, x_cond, x_uncond, t)        

    def reset_partial_pcd(self, x_part, x_uncond, bs):
        x_part = self.points_to_tensor(x_part.F.reshape(bs,-1,3).detach())
        x_uncond = self.points_to_tensor(torch.zeros_like(x_part.F.reshape(bs,-1,3)))

        return x_part, x_uncond  
    
    def validation_step(self, batch):
        
        self.model.eval()
        self.partial_enc.eval()

        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()
            x_0 = batch['pcd_part'].repeat(1,10,1)
            noise = torch.randn(x_0.shape, device=self.device)
            x_0 = x_0 + noise
            x_cond = self.points_to_tensor(batch['pcd_part'])
            x_uncond = self.points_to_tensor(torch.zeros_like(batch['pcd_part']))
            x_gen_eval = self.generate(x_0, x_cond, x_uncond)

            for i in range(len(batch['pcd_full'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()

                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < self.hparams['data']['params']['max_range']
                points = c_pred[dist_idx]
                max_z = batch['pcd_part'][i][...,2].max().item()
                min_z = (batch['pcd_part'][i][...,2].mean() - 2 * batch['pcd_part'][i][...,2].std()).item()
                pcd_pred.points = o3d.utility.Vector3dVector(points[(points[:,2] < max_z) & (points[:,2] > min_z)])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)
    
    def on_validation_epoch_end(self):
        
        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()

        self.log('val/cd_mean', cd_mean, prog_bar=True) 
        self.log('val/cd_std', cd_std, prog_bar=True) 
        self.log('val/precision', pr, prog_bar=True)
        self.log('val/recall', re, prog_bar=True) 
        self.log('val/fscore', f1, prog_bar=True)
        
        self.chamfer_distance.reset()
        self.precision_recall.reset()
        torch.cuda.empty_cache()
        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1}
    
    def valid_paths(self, filenames):
        output_paths = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'

            skip.append(os.path.isfile(f'{seq_dir}/{ply_name}'))
            os.makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')

        return np.all(skip), output_paths
    
    def test_step(self, batch):
        self.model.eval()
        self.partial_enc.eval()
        with torch.no_grad():
            skip, output_paths = self.valid_paths(batch['filename'])

            if skip:
                print(f'Skipping generation from {output_paths[0]} to {output_paths[-1]}') 
                return {'test/cd_mean': 0., 'test/cd_std': 0., 'test/precision': 0., 'test/recall': 0., 'test/fscore': 0.}

            x_init = batch['pcd_part'].repeat(1,10,1)
            noise = torch.randn(x_init.shape, device=self.device)
            x_0 = x_init + noise
            x_cond = self.points_to_tensor(batch['pcd_part'])
            x_uncond = self.points_to_tensor(torch.zeros_like(batch['pcd_part']))
            x_gen_eval = self.generate(x_0, x_cond, x_uncond)

            for i in range(len(batch['pcd_full'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < self.hparams['data']['params']['max_range']
                points = c_pred[dist_idx]
                max_z = x_init[i][...,2].max().item()
                min_z = (x_init[i][...,2].mean() - 2 * x_init[i][...,2].std()).item()
                pcd_pred.points = o3d.utility.Vector3dVector(points[(points[:,2] < max_z) & (points[:,2] > min_z)])
                pcd_pred.paint_uniform_color([1.0, 0.,0.])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.paint_uniform_color([0., 1.,0.])
                
                print(f'Saving {output_paths[i]}')
                o3d.io.write_point_cloud(f'{output_paths[i]}', pcd_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/precision', pr, on_step=True)
        self.log('test/recall', re, on_step=True)
        self.log('test/fscore', f1, on_step=True)
        torch.cuda.empty_cache()
        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        scheduler = {
            'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer], [scheduler]
