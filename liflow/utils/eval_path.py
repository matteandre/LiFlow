import os
import numpy as np
import open3d as o3d
from liflow.utils.metrics import ChamferDistance, PrecisionRecall, CompletionIoU, RMSE 
import tqdm
from natsort import natsorted
from liflow.utils.flow_completion_pipeline import FlowCompletion
from liflow.utils.histogram_metrics import compute_hist_metrics 
import click
import json

completion_iou_flow = CompletionIoU()
rmse_flow = RMSE()
chamfer_distance_flow = ChamferDistance()
precision_recall_flow = PrecisionRecall(0.05,2*0.05,100)

completion_iou_refine = CompletionIoU()
rmse_refine = RMSE()
chamfer_distance_refine = ChamferDistance()
precision_recall_refine= PrecisionRecall(0.05,2*0.05,100)

def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def load_poses(calib_fname, poses_fname):
    if os.path.exists(calib_fname):
        calibration = parse_calibration(calib_fname)
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

    poses_file = open(poses_fname)
    poses = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        if os.path.exists(calib_fname):
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        else:
            poses.append(pose)

    return poses

def get_scan_completion(scan_path, path, diff_completion, max_range):
    pcd_file = os.path.join(path, 'velodyne', scan_path)
    points = np.fromfile(pcd_file, dtype=np.float32)
    points = points.reshape(-1,4)
    dist = np.sqrt(np.sum(points[:,:3]**2, axis=-1))
    input_points = points[dist < max_range, :3]
    
    flow_scan, refine_scan = diff_completion.complete_scan(points)
    pcd_pred_flow = o3d.geometry.PointCloud()
    pcd_pred_flow.points = o3d.utility.Vector3dVector(flow_scan)

    pcd_pred_refine = o3d.geometry.PointCloud()
    pcd_pred_refine.points = o3d.utility.Vector3dVector(refine_scan)

    return pcd_pred_flow, pcd_pred_refine, input_points

def get_ground_truth(pose, cur_scan, seq_map, max_range):
    trans = pose[:-1,-1]
    dist_gt = np.sum((seq_map - trans)**2, axis=-1)**.5
    scan_gt = seq_map[dist_gt < max_range]
    scan_gt = np.concatenate((scan_gt, np.ones((len(scan_gt),1))), axis=-1)
    scan_gt = (scan_gt @ np.linalg.inv(pose).T)[:,:3]
    scan_gt = scan_gt[(scan_gt[:,2] > -4.) & (scan_gt[:,2] < 4.4)]
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(scan_gt)

    cur_pcd = o3d.geometry.PointCloud()
    cur_pcd.points = o3d.utility.Vector3dVector(cur_scan)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cur_pcd, voxel_size=10.)
    in_viewpoint = viewpoint_grid.check_if_included(pcd_gt.points)
    points_gt = np.array(pcd_gt.points)
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt[in_viewpoint])

    return pcd_gt


@click.command()
@click.option('--path', '-p', type=str, default='', help='path to the scan sequence')
@click.option('--max_range', '-m', type=float, default=50, help='max range')
@click.option('--steps', '-s', type=int, default=10, help='number of Euler method steps')
@click.option('--flow', '-f', type=str, help='path to the checkpoint for LiFlow model')
@click.option('--refine', '-r', type=str, help='path to the checkpoint for refinement net')
@click.option('--name', '-n', type=str, default='res_log', help='name of the log file')
def main(path, max_range, steps, flow, refine, name): 
    flow_completion = FlowCompletion(flow, refine, steps)

    poses = load_poses(os.path.join(path, 'calib.txt'), os.path.join(path, 'poses.txt'))
    seq_map = np.load(f'{path}/map_clean.npy')

    jsd_3d_flow = []
    jsd_bev_flow = []

    jsd_3d_refine = []
    jsd_bev_refine = []

    for pose, scan_path in tqdm.tqdm(list(zip(poses, natsorted(os.listdir(f'{path}/velodyne'))))):
        pcd_pred_flow, pcd_pred_refine, cur_scan = get_scan_completion(scan_path, path, flow_completion, max_range)
        pcd_gt = get_ground_truth(pose, cur_scan, seq_map, max_range)

        jsd_3d_flow.append(compute_hist_metrics(pcd_gt, pcd_pred_flow, bev=False))
        jsd_bev_flow.append(compute_hist_metrics(pcd_gt, pcd_pred_flow, bev=True))

        rmse_flow.update(pcd_gt, pcd_pred_flow)
        completion_iou_flow.update(pcd_gt, pcd_pred_flow)
        chamfer_distance_flow.update(pcd_gt, pcd_pred_flow)
        precision_recall_flow.update(pcd_gt, pcd_pred_flow)

        jsd_3d_refine.append(compute_hist_metrics(pcd_gt, pcd_pred_refine, bev=False))
        jsd_bev_refine.append(compute_hist_metrics(pcd_gt, pcd_pred_refine, bev=True))

        rmse_refine.update(pcd_gt, pcd_pred_refine)
        completion_iou_refine.update(pcd_gt, pcd_pred_refine)
        chamfer_distance_refine.update(pcd_gt, pcd_pred_refine)
        precision_recall_refine.update(pcd_gt, pcd_pred_refine)


        print('\n\n=================== FLOW MATCHING ===================\n\n')

        print(f'JSD 3D: {jsd_3d_flow[-1]}')
        print(f'JSD BEV: {jsd_bev_flow[-1]}')

        rmse_mean, rmse_std = rmse_flow.compute()
        print(f'RMSE Mean: {rmse_mean}\tRMSE Std: {rmse_std}')
        thr_ious = completion_iou_flow.compute()
        for v_size in thr_ious.keys():
            print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
        cd_mean, cd_std = chamfer_distance_flow.compute()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        pr, re, f1 = precision_recall_flow.compute_auc()
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')


        print('\n\n=================== REFINEMENT ===================\n\n')

        print(f'JSD 3D: {jsd_3d_refine[-1]}')
        print(f'JSD BEV: {jsd_bev_refine[-1]}')

        rmse_mean, rmse_std = rmse_refine.compute()
        print(f'RMSE Mean: {rmse_mean}\tRMSE Std: {rmse_std}')
        thr_ious = completion_iou_refine.compute()
        for v_size in thr_ious.keys():
            print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
        cd_mean, cd_std = chamfer_distance_refine.compute()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        pr, re, f1 = precision_recall_refine.compute_auc()
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')

    
    res_dict = {}


    print('\n\n=================== FINAL RESULTS ===================\n\n')
    

    print('\n\n=================== FLOW MATCHING ===================\n\n')

    print(f'JSD 3D: {np.array(jsd_3d_flow).mean()}')
    print(f'JSD BEV: {np.array(jsd_bev_flow).mean()}')
    rmse_mean, rmse_std = rmse_flow.compute()
    print(f'RMSE Mean: {rmse_mean}\tRMSE Std: {rmse_std}')
    thr_ious = completion_iou_flow.compute()
    for v_size in thr_ious.keys():
        print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
    cd_mean, cd_std = chamfer_distance_flow.compute()
    print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
    pr, re, f1 = precision_recall_flow.compute_auc()
    print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')
    
    res_dict['flow'] = {
        'jsd': np.array(jsd_bev_flow).mean(),
        'jsd_noclip_3d': np.array(jsd_3d_flow).mean(),
        'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
        'ious': thr_ious,
        'cd_mean': cd_mean, 'cd_std': cd_std,
        'pr': pr, 're': re, 'f1': f1,
    }
    

    print('\n\n=================== REFINEMENT ===================\n\n')
    
    print(f'JSD 3D: {np.array(jsd_3d_refine).mean()}')
    print(f'JSD BEV: {np.array(jsd_bev_refine).mean()}')
    rmse_mean, rmse_std = rmse_refine.compute()
    print(f'RMSE Mean: {rmse_mean}\tRMSE Std: {rmse_std}')
    thr_ious = completion_iou_refine.compute()
    for v_size in thr_ious.keys():
        print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
    cd_mean, cd_std = chamfer_distance_refine.compute()
    print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
    pr, re, f1 = precision_recall_refine.compute_auc()
    print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')
    
    res_dict['refine'] = {
        'jsd': np.array(jsd_bev_refine).mean(),
        'jsd_noclip_3d': np.array(jsd_3d_refine).mean(),
        'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
        'ious': thr_ious,
        'cd_mean': cd_mean, 'cd_std': cd_std,
        'pr': pr, 're': re, 'f1': f1,
    }

    with open(f'{name}.yaml', 'w+') as log_res:
        json.dump(res_dict, log_res)

if __name__ == '__main__':
    main()
