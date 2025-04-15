import pathlib
from typing import Optional
import cv2
import os
import json
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(save_path = 'default', dataset = None):
    save_dir = pathlib.Path("logs_mast3r_slam")
    if save_path != "default":
        save_dir = save_dir / save_path
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        traj = []
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            traj.append(T_WC.data.numpy())
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
        # concat
        traj = np.concatenate(traj, axis=0)
    return traj

def save_traj_full(
    timestamps,
    keyframes: SharedKeyframes,
    Frames: list,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    # logdir = pathlib.Path(logdir)
    # logdir.mkdir(exist_ok=True, parents=True)
    # logfile = logdir / logfile
    # with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
    traj = []
    for i in range(len(Frames)):
        frame = Frames[i]
        t = timestamps[frame.frame_id]
        if intrinsics is None:
            T_WC = as_SE3(frame.T_WC)
        else:
            T_WC = intrinsics.refine_pose_with_calibration(frame)
        x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
        traj.append(T_WC.data.numpy())
    # concat
    traj = np.concatenate(traj, axis=0)
    return traj


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    keyframe_pcd_savedir = pathlib.Path(f"{savedir}/keyframe_pcd/image")
    keyframe_pcd_savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    canon_to_w_scale = {}
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        # valid_output = (valid.astype(np.uint8)) * 255
        # cv2.imwrite(os.path.join(keyframe_pcd_savedir, f"{keyframe.frame_id:05d}_mask.png"), valid_output)
        # NOTE SWH: remove black person masks
        black_mask_flat = np.all(keyframe.uimg.cpu().numpy().reshape(-1, 3) == 0, axis=-1)
        valid = valid & (~black_mask_flat)
        
        # NOTE SWH: ADD pcd mask for scale estimation
        np.save(os.path.join(keyframe_pcd_savedir, f"{keyframe.frame_id:05d}_mask.npy"), valid)
        pointclouds.append(pW[valid])
        colors.append(color[valid])

        save_ply(os.path.join(keyframe_pcd_savedir, f"{keyframe.frame_id:05d}.ply"), pointclouds[-1], colors[-1])
        # save_ply(os.path.join(keyframe_pcd_savedir, f"{keyframe.frame_id:05d}.ply"), pW, color)
        save_ply(os.path.join(keyframe_pcd_savedir, f"{keyframe.frame_id:05d}_canon.ply"), keyframe.X_canon.cpu().numpy().reshape(-1, 3), color)
        C2W_scale = keyframe.T_WC.log()[:, 6].exp().item()
        canon_to_w_scale[keyframe.frame_id] = C2W_scale

    pointclouds_ = np.concatenate(pointclouds, axis=0)
    colors_ = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds_, colors_)

    stats_output_path = pathlib.Path(f"{savedir}/canon_to_w_scale.json")
    with open(stats_output_path, 'w') as f:
        json.dump(canon_to_w_scale, f, indent=4)
    print(f"Saved stats to {stats_output_path}")
    
    return pointclouds_, pointclouds

def save_reconstruction_mono(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    keyframe_pcd_savedir = pathlib.Path(f"{savedir}")
    keyframe_pcd_savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        # pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        # valid = (
        #     keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
        #     > c_conf_threshold
        # )
        # NOTE SWH: remove black person masks
        # black_mask_flat = np.all(keyframe.uimg.cpu().numpy().reshape(-1, 3) == 0, axis=-1)
        # valid = valid & (~black_mask_flat)
        human_mask_flat = np.all(keyframe.mask.cpu().numpy().reshape(-1, 1) == 1, axis=-1)

        save_ply(os.path.join(keyframe_pcd_savedir, f"{keyframe.frame_id:05d}_cam.ply"), keyframe.X_canon.cpu().numpy().reshape(-1, 3), color)
        print(keyframe.X_canon.cpu().numpy().reshape(-1, 3).shape)
        save_ply(os.path.join(keyframe_pcd_savedir, f"{keyframe.frame_id:05d}_human.ply"), keyframe.X_canon.cpu().numpy().reshape(-1, 3)[human_mask_flat], color[human_mask_flat])
        print(keyframe.X_canon.cpu().numpy().reshape(-1, 3)[human_mask_flat].shape)
    
    return


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{keyframe.frame_id:05d}_{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )

def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
