import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import copy
import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob
from tqdm import tqdm
from collections import defaultdict

from lib.utils.eval_utils import *
from lib.utils.rotation_conversions import *
from lib.vis.traj import *
from lib.camera.slam_utils import eval_slam

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--input_dir', type=str, default='./results_debug')
parser.add_argument('--scale', type=float, default=1, help='set the camera translation scale')
parser.add_argument('--vis_figure', type=str, default='./results_debug', help='visualization path figure')
args = parser.parse_args()
input_dir = args.input_dir

# NOTE: emdb seq hard code
#emdb = ['dataset/emdb/P0/09_outdoor_walk', 'dataset/emdb/P2/19_indoor_walk_off_mvs']
#emdb = ['dataset/emdb/P0/09_outdoor_walk', 'dataset/emdb/P2/19_indoor_walk_off_mvs']
# NOTE: emdb seq hard code - For SWH
emdb = [
        '../dataset/P0/09_outdoor_walk',
        # 'dataset/emdb/P2/19_indoor_walk_off_mvs',
        # 'dataset/emdb/P2/20_outdoor_walk',
        # 'dataset/emdb/P2/24_outdoor_long_walk',
        # 'dataset/emdb/P3/27_indoor_walk_off_mvs',
        # 'dataset/emdb/P3/28_outdoor_walk_lunges',
        # 'dataset/emdb/P3/29_outdoor_stairs_up',
        # 'dataset/emdb/P3/30_outdoor_stairs_down',
        # 'dataset/emdb/P4/35_indoor_walk',
        # 'dataset/emdb/P4/36_outdoor_long_walk',
        # 'dataset/emdb/P4/37_outdoor_run_circle',
        # 'dataset/emdb/P5/40_indoor_walk_big_circle',
        # 'dataset/emdb/P6/48_outdoor_walk_downhill',
        # 'dataset/emdb/P6/49_outdoor_big_stairs_down',
        # 'dataset/emdb/P7/55_outdoor_walk',
        # 'dataset/emdb/P7/56_outdoor_stairs_up_down',
        # 'dataset/emdb/P7/57_outdoor_rock_chair',
        # 'dataset/emdb/P7/58_outdoor_parcours',
        # 'dataset/emdb/P7/61_outdoor_sit_lie_walk',
        # 'dataset/emdb/P8/64_outdoor_skateboard',
        # 'dataset/emdb/P8/65_outdoor_walk_straight',
        # 'dataset/emdb/P9/77_outdoor_stairs_up',
        # 'dataset/emdb/P9/78_outdoor_stairs_up_down',
        # 'dataset/emdb/P9/79_outdoor_walk_rectangle',
        # 'dataset/emdb/P9/80_outdoor_walk_big_circle',
        ]


# SMPL
smpl = SMPL()
smpls = {g:SMPL(gender=g) for g in ['neutral', 'male', 'female']}


# Evaluations: world-coordinate SMPL
accumulator = defaultdict(list)
m2mm = 1e3
human_traj = {}
total_invalid = 0

# NOTE SWH: manual set scale for now
scale = args.scale

for root in tqdm(emdb):
    # GT
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))

    ext = ann['camera']['extrinsics']  # in the forms of R_cw, t_cw
    intr = ann['camera']['intrinsics']
    img_focal = (intr[0,0] +  intr[1,1]) / 2.
    img_center = intr[:2, 2]

    valid = ann['good_frames_mask']
    gender = ann['gender']
    poses_body = ann["smpl"]["poses_body"]
    poses_root = ann["smpl"]["poses_root"]
    betas = np.repeat(ann["smpl"]["betas"].reshape((1, -1)), repeats=ann["n_frames"], axis=0)
    trans = ann["smpl"]["trans"]
    total_invalid += (~valid).sum()

    tt = lambda x: torch.from_numpy(x).float()
    gt = smpls[gender](body_pose=tt(poses_body), global_orient=tt(poses_root), betas=tt(betas), transl=tt(trans),
                    pose2rot=True, default_smpl=True)
    gt_vert = gt.vertices
    gt_j3d = gt.joints[:,:24] 
    gt_ori = axis_angle_to_matrix(tt(poses_root))

    # Groundtruth local motion
    poses_root_cam = matrix_to_axis_angle(tt(ext[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root)))
    gt_cam = smpls[gender](body_pose=tt(poses_body), global_orient=poses_root_cam, betas=tt(betas),
                           pose2rot=True, default_smpl=True)
    gt_vert_cam = gt_cam.vertices
    gt_j3d_cam = gt_cam.joints[:,:24]
    
    # PRED
    seq = root.split('/')[-1]
    pred_cam = dict(np.load(f'{input_dir}/{seq}/camera.npy', allow_pickle=True).item())
    # NOTE: check person id
    pred_smpl = dict(np.load(f'{input_dir}/{seq}/hps/hps_track_0.npy', allow_pickle=True).item())

    pred_rotmat = torch.tensor(pred_smpl['pred_rotmat'])
    pred_shape = torch.tensor(pred_smpl['pred_shape'])
    pred_trans = torch.tensor(pred_smpl['pred_trans'])

    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.repeat(len(pred_shape), 1)

    pred = smpls['neutral'](body_pose=pred_rotmat[:,1:], 
                            global_orient=pred_rotmat[:,[0]], 
                            betas=pred_shape, 
                            transl=pred_trans.squeeze(),
                            pose2rot=False, 
                            default_smpl=True)
    pred_vert = pred.vertices
    pred_j3d = pred.joints[:, :24]

    pred_camt = torch.tensor(pred_cam['pred_cam_T']) * scale
    # print(pred_camt.shape)
    pred_camr = torch.tensor(pred_cam['pred_cam_R'])
   
    pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:,None]
    pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:,None]
    pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, pred_rotmat[:,0])
    pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)

    # Valid mask
    gt_j3d = gt_j3d[valid]
    gt_ori = gt_ori[valid]
    pred_j3d_w  = pred_j3d_w[valid]
    pred_ori_w = pred_ori_w[valid]

    gt_j3d_cam = gt_j3d_cam[valid]
    gt_vert_cam = gt_vert_cam[valid]
    pred_j3d = pred_j3d[valid]
    pred_vert = pred_vert[valid]

    # <======= Evaluation on the local motion
    pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam = batch_align_by_pelvis(
        [pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam], pelvis_idxs=[1,2]
    )
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, gt_j3d_cam)
    pa_mpjpe = torch.sqrt(((S1_hat - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
    mpjpe = torch.sqrt(((pred_j3d - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
    pve = torch.sqrt(((pred_vert - gt_vert_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm

    accel = compute_error_accel(joints_pred=pred_j3d.cpu(), joints_gt=gt_j3d_cam.cpu())[1:-1]
    accel = accel * (30 ** 2)       # per frame^s to per s^2

    accumulator['pa_mpjpe'].append(pa_mpjpe)
    accumulator['mpjpe'].append(mpjpe)
    accumulator['pve'].append(pve)
    accumulator['accel'].append(accel)
    # =======>

    # <======= Evaluation on the global motion
    chunk_length = 100
    w_mpjpe, wa_mpjpe = [], []
    for start in range(0, valid.sum() - chunk_length, chunk_length):
        end = start + chunk_length
        if start + 2 * chunk_length > valid.sum(): end = valid.sum() - 1
        
        target_j3d = gt_j3d[start:end].clone().cpu()
        pred_j3d = pred_j3d_w[start:end].clone().cpu()
        
        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)
        
        w_jpe = compute_jpe(target_j3d, w_j3d)
        wa_jpe = compute_jpe(target_j3d, wa_j3d)
        w_mpjpe.append(w_jpe)
        wa_mpjpe.append(wa_jpe)

    w_mpjpe = np.concatenate(w_mpjpe) * m2mm
    wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm
    # =======>

    # <======= Evaluation on the entier global motion
    # RTE: root trajectory error
    pred_j3d_align = first_align_joints(gt_j3d, pred_j3d_w)
    rte_align_first= compute_jpe(gt_j3d[:,[0]], pred_j3d_align[:,[0]])
    rte_align_all = compute_rte(gt_j3d[:,0], pred_j3d_w[:,0]) * 1e2 

    # ERVE: Ego-centric root velocity error
    erve = computer_erve(gt_ori, gt_j3d, pred_ori_w, pred_j3d_w) * m2mm
    # =======>

    # <======= Record human trajectory
    human_traj[seq] = {'gt': gt_j3d[:,0], 'pred': pred_j3d_align[:, 0]}
    # =======>

    accumulator['wa_mpjpe'].append(wa_mpjpe)
    accumulator['w_mpjpe'].append(w_mpjpe)
    accumulator['rte'].append(rte_align_all)
    accumulator['erve'].append(erve)

copied_accumulator = copy.deepcopy(accumulator)

for k, v in accumulator.items():
    accumulator[k] = np.concatenate(v).mean()


# Evaluation: Camera motion
results = {}
for root in emdb:
    # Annotation
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))

    ext = ann['camera']['extrinsics']
    cam_r = ext[:,:3,:3].transpose(0,2,1)
    cam_t = np.einsum('bij, bj->bi', cam_r, -ext[:, :3, -1])
    cam_q = matrix_to_quaternion(torch.from_numpy(cam_r)).numpy()

    # PRED
    seq = root.split('/')[-1]
    pred_cam = dict(np.load(f'{input_dir}/{seq}/camera.npy', allow_pickle=True).item())

    pred_camt = torch.tensor(pred_cam['pred_cam_T']) * scale
    pred_camr = torch.tensor(pred_cam['pred_cam_R'])
    pred_camq = matrix_to_quaternion(pred_camr)
    pred_traj = torch.concat([pred_camt, pred_camq], dim=-1).numpy()

    stats_slam, traj_ref_align, traj_est_align = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=True)
    stats_metric, traj_ref, traj_est = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=False)
    
    # Save results
    re = {'traj_gt': traj_ref.positions_xyz,
          'traj_est': traj_est.positions_xyz, 
          'traj_gt_q': traj_ref.orientations_quat_wxyz,
          'traj_est_q': traj_est.orientations_quat_wxyz,
          'stats_slam': stats_slam,
          'stats_metric': stats_metric}
    
    # Visualize the camera motions
    if args.vis_figure != None:
        plot_trajectories_3d(traj_ref_align.positions_xyz, traj_est_align.positions_xyz, os.path.join(args.vis_figure, seq))
        plot_trajectories_2d(traj_ref_align.positions_xyz, traj_est_align.positions_xyz, os.path.join(args.vis_figure, seq))
    
    results[seq] = re
    copied_accumulator['ate'].append(re['stats_slam']['mean'])
    copied_accumulator['ate_s'].append(re['stats_metric']['mean'])

ate = np.mean([re['stats_slam']['mean'] for re in results.values()])
ate_s = np.mean([re['stats_metric']['mean'] for re in results.values()])
accumulator['ate'] = ate
accumulator['ate_s'] = ate_s

# Save evaluation results
for k, v in accumulator.items():
    print(k, accumulator[k])

df = pd.DataFrame(list(accumulator.items()), columns=['Metric', 'Value'])
df.to_excel(f"{args.input_dir}/evaluation.xlsx", index=False)

excel_rows = []

for i, seq in enumerate(human_traj.keys()):
    print(seq)
    row = {'seq': seq, 'scale': scale}

    # 加入原本accumulator中的指标
    for k in copied_accumulator:
        # print(k)
        val = copied_accumulator[k][i]
        if isinstance(val, np.ndarray):
            row[k] = val.mean() if val.ndim > 0 else val.item()
        elif torch.is_tensor(val):
            row[k] = val.mean().item() if val.ndim > 0 else val.item()
        else:
            row[k] = val

    excel_rows.append(row)

df = pd.DataFrame(excel_rows)
df_numeric = df.drop(columns=['seq'])  # 去掉非数值列
mean_row = df_numeric.mean(numeric_only=True)

# 创建一个 dict，用于添加到 DataFrame 末尾
mean_row = {'seq': 'mean'}
mean_row.update(accumulator)

# 添加到 DataFrame 末尾
df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

excel_path = f'full_evaluation_results_s{scale}.xlsx'
df.to_excel(excel_path, index=False)
print(f"Full evaluation results saved to: {excel_path}")
