import numpy as np
import cv2
import torch
import os
from torchmin import minimize

import trimesh


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def compute_scales(traj, traj_full, pc_whole, pc, kf_idx, smpls=None, smpl_path=None, save_smpl = True, device="cuda:0"):

    torch.cuda.set_device(device)
    with torch.no_grad():
        if isinstance(smpls, dict):
            smpls = [smpls]
        else:
            pass
        #=======================Compute the meteric scales=============================
        traj = torch.tensor(traj).cpu()
        traj_full = torch.tensor(traj_full).cpu()
        kf_len = traj.shape[0]
        kf_idx = kf_idx[:kf_len]
        
        # Find the longest SMPL sequence (multi-person)
        smpl_l = max(smpls, key = lambda x: len(x['pred_verts']) if 'pred_verts' in x else 0)
        # select SMPL attributes in each key frame
        # for k in smpl_l.keys():
        #     if k == 'smpl_faces':
        #         continue
        #     smpl_l[k] = torch.index_select(smpl_l[k], 0, kf_idx.cpu().long()) 

        smpl_l['pred_verts_cam'] = smpl_l['pred_verts'] + smpl_l['pred_trans'] # 2D (SMPL model forward)

        scale = 1 
        
        pred_cam_t = traj_full[:, :3] * scale
        pred_cam_q = traj_full[:, 3:]
        pred_cam_r = quaternion_to_matrix(pred_cam_q[:,[3,0,1,2]])

        pred_vert_w = torch.einsum('bij,bnj->bni', pred_cam_r, smpl_l['pred_verts_cam']) + pred_cam_t[:, None] 

        
        #=======================Compute the meteric scales=============================
        # Save the mesh for visualization
        if save_smpl:
            for track in range(len(smpls)):
                smpl = smpls[track]
                frame_id = smpl['frame']
                smpl['pred_verts_cam'] = smpl['pred_verts'] + smpl['pred_trans']
                
                save_path = os.path.join(smpl_path, str(track))
                os.makedirs(save_path, exist_ok=True)
                for idx in range(smpl['pred_verts_cam'].shape[0]):
                    trimesh.Trimesh(pred_vert_w[idx], smpls[0]['smpl_faces']).export(os.path.join(save_path, f'{frame_id.long()[idx]:05d}.obj'))
                    trimesh.Trimesh(smpl_l['pred_verts_cam'][idx], smpls[0]['smpl_faces']).export(os.path.join(save_path, f'{frame_id.long()[idx]:05d}_cam.obj'))
        return pred_cam_r, pred_cam_t

def est_scale_iterative(slam_depth, pred_depth, iters=10, msk=None):
    """ Simple depth-align by iterative median and thresholding """
    s = pred_depth / slam_depth
    
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    robust = (msk<0.5) * (0<pred_depth) * (pred_depth<10)
    s_est = s[robust]
    scale = np.median(s_est)
    scales_ = [scale]

    for _ in range(iters):
        slam_depth_0 = slam_depth * scale
        robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<10) * (0<pred_depth) * (pred_depth<10)
        s_est = s[robust]
        scale = np.median(s_est)
        scales_.append(scale)

    return scale


def est_scale_gmof(slam_depth, pred_depth, lr=1, sigma=0.5, iters=500, msk=None):
    """ Simple depth-align by robust least-square """
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    robust = (msk<0.5) * (0<pred_depth) * (pred_depth<10)
    pm = torch.from_numpy(pred_depth[robust])
    sm = torch.from_numpy(slam_depth[robust])

    scale = torch.tensor([1.], requires_grad=True)
    optim = torch.optim.Adam([scale], lr=lr)
    losses = []
    for i in range(iters):
        loss = sm * scale - pm
        loss = gmof(loss, sigma=sigma).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    scale = scale.detach().cpu().item()

    return scale


def est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=None, 
                     far_thresh=10):
    """ Depth-align by iterative + robust least-square """
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    # Stage 1: Iterative steps
    s = pred_depth / slam_depth

    robust = (msk<0.5) * (0<pred_depth) * (pred_depth<10)
    s_est = s[robust]
    scale = np.median(s_est)

    for _ in range(10):
        slam_depth_0 = slam_depth * scale
        robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<far_thresh) * (0<pred_depth) * (pred_depth<far_thresh)
        s_est = s[robust]
        scale = np.median(s_est)


    # Stage 2: Robust optimization
    robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<far_thresh) * (0<pred_depth) * (pred_depth<far_thresh)
    pm = torch.from_numpy(pred_depth[robust])
    sm = torch.from_numpy(slam_depth[robust])

    def f(x):
        loss = sm * x - pm
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([scale])
    result = minimize(f, x0,  method='bfgs')
    scale = result.x.detach().cpu().item()

    return scale


def scale_shift_align(smpl_depth, pred_depth, sigma=0.5):
    """ Align pred_depth to smpl depth """
    smpl = torch.from_numpy(smpl_depth)
    pred = torch.from_numpy(pred_depth)

    def f(x):
        loss = smpl - (pred * x[0] + x[1])
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([1., 0.])
    result = minimize(f, x0,  method='bfgs')
    scale_shift = result.x.detach().cpu().numpy()

    return scale_shift


def shift_align(smpl_depth, pred_depth, sigma=0.5):
    """ Align pred_depth to smpl depth by only shift """
    smpl = torch.from_numpy(smpl_depth)
    pred = torch.from_numpy(pred_depth)

    def f(x):
        loss = smpl - (pred + x)
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([0.])
    result = minimize(f, x0,  method='bfgs')
    scale_shift = result.x.detach().cpu().numpy()

    return scale_shift


def gmof(x, sigma=100):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

