import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from scipy.ndimage import gaussian_filter

from .tools import checkerboard_geometry
from lib.models.smpl import SMPL
from lib.utils.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np


def traj_filter(pred_vert_w, pred_j3d_w, sigma=3):
    """ Smooth the root trajetory (xyz) """
    root = pred_j3d_w[:, 0]
    root_smooth = torch.from_numpy(gaussian_filter(root, sigma=sigma, axes=0))

    pred_vert_w = pred_vert_w + (root_smooth - root)[:, None]
    pred_j3d_w = pred_j3d_w + (root_smooth - root)[:, None]
    return pred_vert_w, pred_j3d_w

def cam_filter(cam_r, cam_t, r_sigma=3, t_sigma=15):
    """ Smooth camera trajetory (SO3) """
    cam_q = matrix_to_quaternion(cam_r)
    r_smooth = torch.from_numpy(gaussian_filter(cam_q, sigma=r_sigma, axes=0))
    t_smooth = torch.from_numpy(gaussian_filter(cam_t, sigma=t_sigma, axes=0))

    r_smooth = r_smooth / r_smooth.norm(dim=1, keepdim=True)
    r_smooth = quaternion_to_matrix(r_smooth)
    return r_smooth,  t_smooth

def fit_to_ground_easy(pred_vert_w, pred_j3d_w, idx=-1):
    """
    Transform meshes to a y-up ground plane
    pred_vert_w (B, N, 3)
    pred_j3d_w (B, J, 3)
    """
    # fit a ground plane
    toes = pred_j3d_w[:, [10, 11]]
    toes = toes.reshape(1, -1, 3)
    pl = fit_plane(toes, idx)

    normal = pl[0, :3]
    offset = pl[0, -1]
    person_up = (pred_j3d_w[:, 3] - pred_j3d_w[:, 0]).mean(dim=0)
    if (person_up @ normal).sign() < 0:
        normal = -normal
        offset = -offset

    yup = torch.tensor([0, 1., 0])
    R = align_a2b(normal, yup)

    pred_vert_gr = torch.einsum('ij,bnj->bni', R, pred_vert_w)
    pred_j3d_gr = torch.einsum('ij,bnj->bni', R, pred_j3d_w)
    offset = pred_vert_gr[:, :, 1].min()

    return R, offset

def fit_to_ground_spine(pred_vert_w, pred_j3d_w, start=0, end=15, lowest=None):
    """
    Transform to a y-up ground plane using the spine direction
    pred_vert_w (B, N, 3)
    pred_j3d_w (B, J, 3)
    """
    # fit a ground plane
    person_up = (pred_j3d_w[start:end, 6] - pred_j3d_w[start:end, 3]).mean(dim=0)
    person_up /= person_up.norm()
    yup = torch.tensor([0, 1., 0])
    R = align_a2b(person_up, yup)

    pred_vert_gr = torch.einsum('ij,bnj->bni', R, pred_vert_w)
    pred_j3d_gr = torch.einsum('ij,bnj->bni', R, pred_j3d_w)

    if lowest is None:
        lowest = end
    offset = pred_vert_gr[0:lowest, :, 1].min()
    
    pred_vert_gr[...,1] -= offset
    pred_j3d_gr[...,1] -= offset

    return pred_vert_gr, pred_j3d_gr

def fit_plane(points, idx=-1):
    """
    From SLAHMR
    :param points (*, N, 3)
    returns (*, 3) plane parameters (returns in normal * offset format)
    """
    *dims, N, D = points.shape
    mean = points.mean(dim=-2, keepdim=True)
    # (*, N, D), (*, D), (*, D, D)
    U, S, Vh = torch.linalg.svd(points - mean)
    normal = Vh[..., idx, :]  # (*, D)
    offset = torch.einsum("...ij,...j->...i", points, normal)  # (*, N)
    offset = offset.mean(dim=-1, keepdim=True)
    return torch.cat([normal, offset], dim=-1)

def get_floor_mesh(pred_vert_gr, z_start=0, z_end=-1, scale=1.5):
    """ Return the geometry of the floor mesh """
    verts = pred_vert_gr.clone()

    # Scale of the scene
    sx, sz = (verts.mean(1).max(0)[0] - verts.mean(1).min(0)[0])[[0, 2]]
    scale = max(sx.item(), sz.item()) * scale

    # Center X
    cx = (verts.mean(1).max(0)[0] + verts.mean(1).min(0)[0])[[0]] / 2.0
    cx = cx.item()

    # Center Z: optionally using only a subsection
    verts = verts[z_start:z_end]
    cz = (verts.mean(1).max(0)[0] + verts.mean(1).min(0)[0])[[2]] / 2.0
    cz = cz.item()

    v, f, vc, fc = checkerboard_geometry(length=scale, c1=cx, c2=cz, up="y")
    vc = vc[:, :3] * 255
  
    return [v, f, vc]

def get_mesh_world(pred_smpl, pred_cam=None, scale=None):
    """ Transform smpl from canonical to world frame """
    smpl = SMPL()

    pred_rotmat = pred_smpl['pred_rotmat']
    pred_shape = pred_smpl['pred_shape']
    pred_trans = pred_smpl['pred_trans']

    pred = smpl(body_pose=pred_rotmat[:,1:], 
                global_orient=pred_rotmat[:,[0]], 
                betas=pred_shape, 
                transl=pred_trans.squeeze(),
                pose2rot=False, 
                default_smpl=True)
    pred_vert = pred.vertices
    pred_j3d = pred.joints[:, :24]

    if pred_cam is not None:
        pred_traj = pred_cam['traj']
        pred_camt = torch.tensor(pred_traj[:, :3]) * scale
        pred_camq = torch.tensor(pred_traj[:, 3:])
        pred_camr = quaternion_to_matrix(pred_camq[:,[3,0,1,2]])

        pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:,None]
        pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:,None]
    else:
        pred_vert_w = pred_vert
        pred_j3d_w = pred_j3d

    return pred_vert_w, pred_j3d_w
    
def align_a2b(a, b):
    # Find a rotation that align a to b
    v = torch.cross(a, b)
    # s = v.norm()
    c = torch.dot(a, b)
    R = torch.eye(3) + skew(v) + skew(v)@skew(v) * (1/(1+c))
    return R

def skew(a):
    v1, v2, v3 = a
    m = torch.tensor([[0, -v3, v2],
                      [v3, 0, -v1],
                      [-v2, v1, 0]]).float()
    return m

def vis_traj(traj_1, traj_2, savefolder, grid=5):
    """ Plot & compare the trajetories in the xy plane """
    os.makedirs(savefolder, exist_ok=True)

    for seq in traj_1:
        traj_gt = traj_1[seq]['gt']
        traj_1 = traj_1[seq]['pred']
        traj_w = traj_2[seq]['pred']

        vis_center = traj_gt[0]
        traj_1 = traj_1 - vis_center
        traj_w = traj_w - vis_center
        traj_gt = traj_gt - vis_center
        
        length = len(traj_gt)
        step = int(length/100)

        a1 = np.linspace(0.3, 0.90, len(traj_gt[0::step,0]))
        a2 = np.linspace(0.3, 0.90, len(traj_w[0::step,0]))

        plt.rcParams['figure.figsize']=4,3
        fig, ax = plt.subplots()
        colors = ['tab:green', 'tab:blue', 'tab:orange']
        ax.scatter(traj_gt[0::step,0], traj_gt[0::step,2], s=10, c='tab:grey', alpha=a1, edgecolors='none')
        ax.scatter(traj_w[0::step,0], traj_w[0::step,2], s=10, c='tab:blue', alpha=a2, edgecolors='none')
        ax.scatter(traj_1[0::step,0], traj_1[0::step,2], s=10, c='tab:orange', alpha=a1, edgecolors='none')
        ax.set_box_aspect(1)
        ax.set_aspect(1, adjustable='datalim')
        ax.grid(linewidth=0.4, linestyle='--')

        ax.tick_params(axis='both', labelsize=8)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(grid)) 
        fig.savefig(f'{savefolder}/{seq}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

def plot_trajectories_3d(traj1, traj2, save_path, t1_name=None, t2_name=None, fig_name=None):
    """
    可视化两段三维相机轨迹
    :param traj1: 轨迹1的xyz坐标数组 形状(N,3)
    :param traj2: 轨迹2的xyz坐标数组 形状(M,3) 
    :param save_path: 图片保存路径
    """
    # 设置Seaborn样式
    sns.set_style("whitegrid")
    
    # 创建3D画布
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹（调整颜色和线宽）
    ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], 
            color=sns.color_palette("tab10")[0], 
            linewidth=2, 
            label=t1_name if t1_name != None else 'GT')
    ax.plot(traj2[:,0], traj2[:,1], traj2[:,2],
            color=sns.color_palette("tab10")[1],
            linewidth=2,
            linestyle='--',
            label=t2_name if t2_name != None else 'Pred')
    
    # 设置观察角度
    ax.view_init(elev=20, azim=45)
    
    # 添加标签和标题
    ax.set_xlabel('X-axis (m)', fontsize=12)
    ax.set_ylabel('Y-axis (m)', fontsize=12)
    ax.set_zlabel('Z-axis (m)', fontsize=12)
    ax.set_title('Traj Comparison', fontsize=16, pad=20)
    
    # 添加图例和网格
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True)
    
    # # 保持横纵比一致
    # max_range = np.array([traj1.max(), traj2.max()]).max()
    # min_val = np.array([traj1.min(), traj2.min()]).min()
    # ax.set_xlim(min_val, max_range)
    # ax.set_ylim(min_val, max_range)
    # ax.set_zlim(min_val, max_range)
    # 组合两个轨迹用于统一范围计算
    all_points = np.vstack([traj1, traj2])
    x_range = all_points[:, 0]
    y_range = all_points[:, 1]
    z_range = all_points[:, 2]

    # 找到中心点和最大跨度
    x_center = (x_range.max() + x_range.min()) / 2
    y_center = (y_range.max() + y_range.min()) / 2
    z_center = (z_range.max() + z_range.min()) / 2
    max_range = max(x_range.ptp(), y_range.ptp(), z_range.ptp()) / 2  # ptp: max - min

    # 设置三轴坐标范围，保持等比例缩放
    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(y_center - max_range, y_center + max_range)
    ax.set_zlim(z_center - max_range, z_center + max_range)
    
    # 保存图像
    if fig_name is None:
        fig_name = 'traj_figure_3d'
    plt.savefig(f'{save_path}/{fig_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_trajectories_2d(traj1, traj2, save_path, plane='xy', t1_name=None, t2_name=None, fig_name=None):
    """
    二维轨迹可视化函数
    :param traj1: 轨迹1的xyz坐标数组 形状(N,3)
    :param traj2: 轨迹2的xyz坐标数组 形状(M,3)
    :param save_path: 图片保存路径
    :param plane: 投影平面选择，可选'xy'(默认)/'xz'/'yz'
    """
    # 设置Seaborn样式[1,6](@ref)
    sns.set_style("whitegrid")

    # 创建画布
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 坐标轴选择
    plane_dict = {
        'xy': (0, 1, 'X-axis (m)', 'Y-axis (m)'),
        'xz': (0, 2, 'X-axis (m)', 'Z-axis (m)'), 
        'yz': (1, 2, 'Y-axis (m)', 'Z-axis (m)')
    }
    idx_x, idx_y, xlabel, ylabel = plane_dict[plane.lower()]

    # 绘制轨迹[6](@ref)
    ax.plot(traj1[:, idx_x], traj1[:, idx_y],
            color=sns.color_palette("tab10")[0],
            linewidth=2,
            linestyle='-',
            label=t1_name if t1_name != None else 'GT')
    
    ax.plot(traj2[:, idx_x], traj2[:, idx_y],
            color=sns.color_palette("tab10")[1],
            linewidth=2,
            linestyle='--',
            label=t2_name if t2_name != None else 'Pred')

    # 坐标轴设置[6](@ref)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Traj Comparison({} plane)'.format(plane.upper()), fontsize=16)
    
    # # 自动适配坐标范围
    # min_val = min(traj1.min(), traj2.min())
    # max_val = max(traj1.max(), traj2.max())
    # ax.set_xlim(min_val, max_val)
    # ax.set_ylim(min_val, max_val)
    # 自动适配坐标范围（按选中坐标轴）
    all_points = np.vstack([traj1, traj2])
    x_vals = all_points[:, idx_x]
    y_vals = all_points[:, idx_y]

    x_center = (x_vals.max() + x_vals.min()) / 2
    y_center = (y_vals.max() + y_vals.min()) / 2
    max_range = max(x_vals.ptp(), y_vals.ptp()) / 2  # ptp: peak-to-peak, max - min

    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(y_center - max_range, y_center + max_range)

    
    # 添加辅助元素[3](@ref)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    if fig_name is None:
        fig_name = 'traj_figure_2d'
    plt.savefig(f'{save_path}/{fig_name}_{plane}.png', dpi=300, bbox_inches='tight')
    plt.close()