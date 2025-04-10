import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import argparse
import numpy as np
from glob import glob
from pycocotools import mask as masktool

from lib.pipeline import video2frames, detect_segment_track, visualize_tram
from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world, run_smpl_metric_slam, run_smpl_metric_slam_mast3r
from lib.camera import run_mast3r_metric_slam
from lib.utils.imutils import copy_images
import pickle as pkl
from tqdm import tqdm

import cv2
from lib.models import get_hmr_vimo
from lib.pipeline import visualize_tram

import warnings
warnings.filterwarnings("ignore")

def write_main_mask(input_folder, output_folder):
    # input_folder = 'results/19_indoor_walk_off_mvs/Annotations'
    # output_folder = 'results/19_indoor_walk_off_mvs/Annotations_main'

    os.makedirs(output_folder, exist_ok=True)

    # Step 1: 读取第一帧，找一个非黑的颜色作为目标颜色
    first_frame_path = os.path.join(input_folder, sorted(os.listdir(input_folder))[0])
    first_frame = cv2.imread(first_frame_path)

    # 找到一个非黑色像素作为目标颜色
    non_black = np.where(np.any(first_frame != [0, 0, 0], axis=-1))
    if len(non_black[0]) == 0:
        raise ValueError("第一帧中没有非背景像素")

    y0, x0 = non_black[0][0], non_black[1][0]
    target_color = first_frame[y0, x0].tolist()

    print(f"目标颜色为: {target_color}")

    # Step 2: 遍历所有帧并提取该颜色的mask
    for fname in tqdm(sorted(os.listdir(input_folder))):
        if not fname.endswith(('.png', '.jpg')): continue

        img = cv2.imread(os.path.join(input_folder, fname))

        # 创建mask：像素值与目标颜色一致的区域设为255，其余为0
        mask = cv2.inRange(img, np.array(target_color), np.array(target_color))

        # 如果整张图都没找到目标颜色，则更新 target_color 为当前帧第一个非黑色像素
        if cv2.countNonZero(mask) == 0:
            non_black = np.where(np.any(img != [0, 0, 0], axis=-1))
            if len(non_black[0]) > 0:
                y0, x0 = non_black[0][0], non_black[1][0]
                target_color = img[y0, x0].tolist()
                print(f"[{fname}] 更新目标颜色为: {target_color}")

                # 重新提取新的颜色区域
                mask = cv2.inRange(img, np.array(target_color), np.array(target_color))
            else:
                print(f"[{fname}] 找不到任何非黑像素，跳过该帧")
                continue

        out_path = os.path.join(output_folder, fname)
        cv2.imwrite(out_path, mask)


def main(args):
    # File and folders
    file = args.input
    root = os.path.normpath(file)
    seq = os.path.basename(root).split('.')[0]

    seq_folder = os.path.join(args.output_dir, seq)
    img_folder = f'{seq_folder}/images'
    hps_folder = f'{seq_folder}/hps'
    smpl_folder = f'{seq_folder}/smpls'
    sam_main_folder = f'{seq_folder}/Annotations_main'
    sam_all_folder = f'{seq_folder}/Annotations'
    
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(hps_folder, exist_ok=True)
    os.makedirs(smpl_folder, exist_ok=True)
    
    # We follow Tram so we use annotated intrinsics and bb for emdb evaluation
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))
    intr = ann['camera']['intrinsics']
    ann_boxes = ann['bboxes']['bboxes']
    cam_int = [intr[0,0], intr[1,1], intr[0,2], intr[1,2]]
    
    ##### Extract Frames #####
    print('Extracting frames ...')
    if file.endswith('.mov') or file.endswith('.mp4'):
        os.makedirs(img_folder, exist_ok=True)
        nframes = video2frames(file, img_folder)
    else:
        copy_images(file, img_folder)

    ##### Detection + SAM + DEVA-Track-Anything #####
    print('Detect, Segment, and Track ...')
    if os.path.exists(f'{seq_folder}/mask.npy'):
        print('Found existing mask.npy, restoring...')
        imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
        masks_ = np.load(f'{seq_folder}/mask.npy', allow_pickle=True)
        tracks_ = np.load(f'{seq_folder}/tracks.npy', allow_pickle=True)
    else:
        imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

        boxes_, masks_, tracks_, _ = detect_segment_track(imgfiles, seq_folder, thresh=0.25, 
                                                          min_size=100, save_vos=args.visualize_mask)
        np.save(f'{seq_folder}/tracks.npy', tracks_)
        np.save(f'{seq_folder}/mask.npy', masks_)
    
    ##### Run Masked Mast3r-SLAM #####
    print('Masked Metric SLAM ...')
    masks = np.array([masktool.decode(m) for m in masks_])
    masks = torch.from_numpy(masks)

    traj, traj_full, pc_whole, pc, kf_idx = run_mast3r_metric_slam(img_folder, masks, cam_int, seq)

    #==========================================================
    if os.path.exists(f'{hps_folder}/hps_track_0.npy'): # We have at least one
        print('Found existing hps sequences, restoring...')
        hps_list = os.listdir(hps_folder)
        hps_list.sort()
        results = []
        for hps_file in hps_list:
            results.append(np.load(os.path.join(hps_folder, hps_file), allow_pickle=True).item())
            
    else:
        print('Estimate HPS ...')
        model = get_hmr_vimo(checkpoint='data/pretrain/vimo_checkpoint.pth.tar')
        
        if args.multiperson:
            tracks = np.load(f'{seq_folder}/tracks.npy', allow_pickle=True).item()
            tid = [k for k in tracks.keys()]
            lens = [len(trk) for trk in tracks.values()]
            rank = np.argsort(lens)[::-1]
            tracks = [tracks[tid[r]] for r in rank]
         
            results_persons = []
            for k, trk in enumerate(tracks):
                valid = np.array([t['det'] for t in trk])
                boxes = np.concatenate([t['det_box'] for t in trk])
                frame = np.array([t['frame'] for t in trk])
                results = model.inference(imgfiles, boxes, valid=valid, frame=frame,
                                          img_focal=cam_int[0], img_center=cam_int[2:])
                
                if results is not None:
                    results_persons.append(results)
                    np.save(f'{hps_folder}/hps_track_{k}.npy', results)
                
                if k+1 >= args.max_humans:
                    break

        else:
            results = model.inference_chunk_emdb(imgfiles, ann_boxes, img_focal=cam_int[0], 
                                                 img_center=cam_int[2:])
        
            np.save(f'{hps_folder}/hps_track_0.npy', results)

    #==========================================================
    cam_R, cam_T = run_smpl_metric_slam_mast3r(traj, traj_full, pc_whole, pc, kf_idx, smpls=results, smpl_path=smpl_folder)

    wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)

    camera = {'pred_cam_R': cam_R.numpy(), 'pred_cam_T': cam_T.numpy(), 
            'world_cam_R': wd_cam_R.numpy(), 'world_cam_T': wd_cam_T.numpy(),
            'img_focal': cam_int[0], 'img_center': cam_int[2:], 'spec_focal': spec_f}

    np.save(f'{seq_folder}/camera.npy', camera)

    if args.visualize_mask:
        write_main_mask(sam_all_folder, sam_main_folder)


    ##### EMDB Evaluation #####


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='./dataset/P9/79_outdoor_walk_rectangle', help='path to your EMDB Test Samples')
    parser.add_argument("--static_camera", action='store_true', help='whether the camera is static')
    parser.add_argument("--visualize_mask", action='store_true', help='save deva vos for visualization')
    parser.add_argument('--max_humans', type=int, default=20, help='maximum number of humans to reconstruct')
    parser.add_argument('--output_dir', type=str, default='results', help='the output save directory')
    parser.add_argument('--bin_size', type=int, default=-1, help='rasterization bin_size; set to [64,128,...] to increase speed')
    parser.add_argument('--floor_scale', type=int, default=3, help='size of the floor')
    parser.add_argument('--multiperson', type=bool, default=False, help='Multi tracks, if you use emdb for evaluation, please set false')
    args = parser.parse_args()
    main(args)