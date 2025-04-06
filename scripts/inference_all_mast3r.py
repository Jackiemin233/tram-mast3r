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

import cv2

from lib.models import get_hmr_vimo
from lib.pipeline import visualize_tram

import warnings
warnings.filterwarnings("ignore")


def main(args):
    # File and folders
    file = args.input
    root = os.path.dirname(file)
    seq = os.path.basename(file).split('.')[0]

    seq_folder = f'results/{seq}'
    img_folder = f'{seq_folder}/images'
    hps_folder = f'{seq_folder}/hps'
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)

    ##### Extract Frames #####
    print('Extracting frames ...')
    if file.endswith('.mov') or file.endswith('.mp4'):
        nframes = video2frames(file, img_folder)
    else:
        pass
        copy_images(file, img_folder)

    ##### Detection + SAM + DEVA-Track-Anything #####
    print('Detect, Segment, and Track ...')
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    #imgfiles = os.listdir()
    # boxes_, masks_, tracks_, bgmasks_ = detect_segment_track(imgfiles, seq_folder, thresh=0.25, 
    #                                                         min_size=100, save_vos=args.visualize_mask)
    # np.save(f'{seq_folder}/tracks.npy', tracks_)
    # np.save(f'{seq_folder}/mask.npy', masks_)
    ##### Run Masked DROID-SLAM #####
    print('Masked Metric SLAM ...')

    masks_ = np.load(f'{seq_folder}/mask.npy', allow_pickle=True)
    masks = np.array([masktool.decode(m) for m in masks_])
    masks = torch.from_numpy(masks)
    #NOTE: NJ modify the bg masks
    # bgmasks = np.array([masktool.decode(m) for m in bgmasks_])
    # bgmasks = torch.from_numpy(bgmasks)
    #cv2.imwrite('/hpc2hdd/home/gzhang292/nanjie/project4/tram/vis/bgmask.png', bgmasks[0].numpy() * 255)

    traj, pc_whole, pc, kf_idx = run_mast3r_metric_slam(img_folder, masks)

    # Calibrate the intrinsics - NJ Step1
    cam_int, _ = calibrate_intrinsics(img_folder, masks, is_static=args.static_camera) # camera intrinsics
    #==========================================================
    # Utilize the intrinsics in HPS methods - NJ Step2
    tracks = np.load(f'{seq_folder}/tracks.npy', allow_pickle=True).item()
    tid = [k for k in tracks.keys()]
    lens = [len(trk) for trk in tracks.values()]
    rank = np.argsort(lens)[::-1]
    tracks = [tracks[tid[r]] for r in rank]

    print('Estimate HPS ...')
    model = get_hmr_vimo(checkpoint='data/pretrain/vimo_checkpoint.pth.tar')

    results_persons = []
    for k, trk in enumerate(tracks): #6 persons in demo in total
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

    #==========================================================
    cam_R, cam_T = run_smpl_metric_slam_mast3r(traj, pc_whole, pc, kf_idx, smpls=results_persons)

    wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)

    camera = {'pred_cam_R': cam_R.numpy(), 'pred_cam_T': cam_T.numpy(), 
            'world_cam_R': wd_cam_R.numpy(), 'world_cam_T': wd_cam_T.numpy(),
            'img_focal': cam_int[0], 'img_center': cam_int[2:], 'spec_focal': spec_f}

    np.save(f'{seq_folder}/camera.npy', camera)
    np.save(f'{seq_folder}/boxes.npy', boxes_)
    np.save(f'{seq_folder}/masks.npy', masks_)
    np.save(f'{seq_folder}/tracks.npy', tracks_)

    # Stage2
    # camera = np.load(f'{seq_folder}/camera.npy', allow_pickle=True).item()
    # tracks = np.load(f'{seq_folder}/tracks.npy', allow_pickle=True).item()

    # img_focal = camera['img_focal']
    # img_center = camera['img_center']

    # tid = [k for k in tracks.keys()]
    # lens = [len(trk) for trk in tracks.values()]
    # rank = np.argsort(lens)[::-1]
    # tracks = [tracks[tid[r]] for r in rank]

    # print('Estimate HPS ...')
    # model = get_hmr_vimo(checkpoint='data/pretrain/vimo_checkpoint.pth.tar')

    # for k, trk in enumerate(tracks):
    #     valid = np.array([t['det'] for t in trk])
    #     boxes = np.concatenate([t['det_box'] for t in trk])
    #     frame = np.array([t['frame'] for t in trk])
    #     results = model.inference(imgfiles, boxes, valid=valid, frame=frame,
    #                               img_focal=img_focal, img_center=img_center)
        
    #     if results is not None:
    #         np.save(f'{hps_folder}/hps_track_{k}.npy', results)
        
    #     if k+1 >= args.max_humans:
    #         break

    ##### Combine camera & human motion #####
    # Render video
    print('Visualize results ...')
    visualize_tram(seq_folder, floor_scale=args.floor_scale, bin_size=args.bin_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='./example_video.mov', help='input video')
    parser.add_argument("--static_camera", action='store_true', help='whether the camera is static')
    parser.add_argument("--visualize_mask", action='store_true', help='save deva vos for visualization')
    parser.add_argument('--max_humans', type=int, default=20, help='maximum number of humans to reconstruct')
    args = parser.parse_args()
    main(args)
