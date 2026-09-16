import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
from pytorch3d.loss import chamfer_distance
import liflow.models.minkunet as minknet
from liflow.utils.helpers import instantiate_from_config
from liflow.utils.collations import *
from liflow.utils.metrics import ChamferDistance, PrecisionRecall

class RefineFlow(LightningModule):
    def __init__(self, resolution, num_points, up_factor, scan_window, lr):
        super().__init__()

        self.lr = lr
        self.resolution = resolution
        self.num_points = num_points
        self.up_factor = up_factor
        self.scan_window = scan_window

        # learn N offsets per point: out_channel is 3 * N
        self.model_refine = minknet.MinkUNet(in_channels=3, out_channels=3*self.up_factor)

        n_part = int(self.num_points / self.scan_window)
        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(0.001,0.01,100)

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

    def forward_refine(self, x):
        return self.model_refine(x)

    def training_step(self, batch, batch_idx):
        x_feats = ME.utils.batched_coordinates(list(batch['pcd_noise']), dtype=torch.float32, device=self.device)
        x_coord = x_feats.clone()
        x_coord = torch.round(x_feats / self.resolution)

        x_feats = x_feats[:,1:]

        x_t = ME.TensorField(
            features=x_feats,
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        offset = self.forward_refine(x_t).reshape(-1,self.up_factor,3)
        refine_upsample_pcd = x_feats[:,None,:] + offset
        refine_upsample_pcd = refine_upsample_pcd.reshape(batch['pcd_full'].shape[0],-1,3)

        loss, _ = chamfer_distance(refine_upsample_pcd, torch.tensor(batch['pcd_full']))
        self.log('train/cd_loss', loss)
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x_feats = ME.utils.batched_coordinates(list(batch['pcd_noise']), dtype=torch.float32, device=self.device)
            x_coord = x_feats.clone()
            x_coord = torch.round(x_feats / self.resolution)
    
            x_feats = x_feats[:,1:]
    
            x_t = ME.TensorField(
                features=x_feats,
                coordinates=x_coord,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=self.device,
            )
    
            offset = self.forward_refine(x_t).reshape(-1,self.up_factor,3)
            refine_upsample_pcd = x_feats[:,None,:] + offset
            refine_upsample_pcd = refine_upsample_pcd.reshape(batch['pcd_full'].shape[0],-1,3)
    
            loss, _ = chamfer_distance(refine_upsample_pcd, torch.tensor(batch['pcd_full']))
            self.log('val/cd_loss', loss)
            torch.cuda.empty_cache()
    
            return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x_feats = ME.utils.batched_coordinates(list(batch['pcd_noise']), dtype=torch.float32, device=self.device)
            x_coord = x_feats.clone()
            x_coord = torch.round(x_feats / self.resolution)
    
            x_feats = x_feats[:,1:]
    
            x_t = ME.TensorField(
                features=x_feats,
                coordinates=x_coord,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=self.device,
            )
    
            offset = self.forward_refine(x_t).reshape(-1,self.up_factor,3)
            refine_pcd = x_feats[:,None,:] + offset
            refine_pcd = refine_pcd.reshape(batch['pcd_full'].shape[0],-1,3)

            pcd_refine = o3d.geometry.PointCloud()
            pcd_refine.points = o3d.utility.Vector3dVector(refine_pcd[0].cpu().numpy())
            pcd_refine.paint_uniform_color([1.,.2,.2])
            pcd_refine.estimate_normals()
            o3d.visualization.draw_geometries([pcd_refine])
    
            loss, _ = chamfer_distance(refine_pcd, torch.tensor(batch['pcd_full']))
            self.log('test/cd_loss', loss)
            torch.cuda.empty_cache()

            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))

        return optimizer
