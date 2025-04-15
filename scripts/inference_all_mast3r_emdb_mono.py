import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import argparse
import numpy as np
from glob import glob
from pycocotools import mask as masktool

from lib.pipeline import video2frames, detect_segment_track
from lib.camera import  align_cam_to_world, run_smpl_metric_slam_mast3r
from lib.camera import run_mast3r_metric_slam, run_mast3r_single_frame
from lib.utils.imutils import copy_images, write_main_mask
import pickle as pkl
from tqdm import tqdm

import cv2
from lib.models import get_hmr_vimo
from lib.pipeline import visualize_tram

import warnings
warnings.filterwarnings("ignore")

def main(args):
    # File and folders
    file = args.input
    root = os.path.normpath(file)
    seq = os.path.basename(root).split('.')[0]

    seq_folder = os.path.join(args.mast3r_output, seq)
    img_folder = f'{seq_folder}/images'
    sam_main_folder = f'{seq_folder}/Annotations_main'
    sam_all_folder = f'{seq_folder}/Annotations'
    
    os.makedirs(seq_folder, exist_ok=True)
    
    # We follow Tram so we use annotated intrinsics and bb for emdb evaluation
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))
    intr = ann['camera']['intrinsics']
    cam_int = [intr[0,0], intr[1,1], intr[0,2], intr[1,2]]

    # NOTE hard code
    keyframe_dir = f"./logs_mast3r_slam/{seq}/keyframes/images"

    # 获取 ID 列表
    keyframe_paths = glob(os.path.join(keyframe_dir, "*.png"))
    ids = set(os.path.basename(p).split("_")[0] for p in keyframe_paths)

    # 目标尺寸
    target_size = (384, 512)  # (width, height)

    # 复制并 resize
    for id in sorted(ids):
        jpg_name = id + ".jpg"
        src_path = os.path.join(img_folder, jpg_name)

        if os.path.exists(src_path):
            img = cv2.imread(src_path)
            mask = np.ones_like(img[..., 0], dtype=np.uint8)
            pc_whole, pc = run_mast3r_single_frame(img, mask=mask, calib=cam_int, seq=seq, image_idx=id, save_dir=args.output_dir)

    #==========================================================

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='./dataset/P9/79_outdoor_walk_rectangle', help='path to your EMDB Test Samples')
    parser.add_argument("--mast3r_output", type=str, default='./results', help='path to your MASt3r results')
    parser.add_argument('--output_dir', type=str, default='results_mono', help='the output save directory')
    args = parser.parse_args()
    main(args)