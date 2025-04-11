import os
import cv2
from glob import glob
import numpy as np
import argparse
import json


def resize_and_copy_keyframes(seq):
    keyframe_dir = f"./logs_mast3r_slam/{seq}/keyframes/images"
    source_jpg_dir = f"./results/{seq}/images"
    target_dir = f"/home/shenwenhao/MoGe/example_images/{seq}"

    print(f"COPY from {source_jpg_dir} to {target_dir}")
    # 确保目标路径存在
    os.makedirs(target_dir, exist_ok=True)

    # 获取 ID 列表
    keyframe_paths = glob(os.path.join(keyframe_dir, "*.png"))
    ids = set(os.path.basename(p).split("_")[0] for p in keyframe_paths)

    # 目标尺寸
    target_size = (384, 512)  # (width, height)

    # 复制并 resize
    for id in ids:
        jpg_name = id + ".jpg"
        src_path = os.path.join(source_jpg_dir, jpg_name)
        dst_path = os.path.join(target_dir, jpg_name)

        if os.path.exists(src_path):
            img = cv2.imread(src_path)
            if img is not None:
                resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst_path, resized_img)
                # print(f"Resized & Copied: {src_path} -> {dst_path}")
            else:
                print(f"Warning: Failed to read {src_path}")
        else:
            print(f"Warning: {src_path} not found.")

def compute_fov_x(seq):
    K = np.load(f'logs_mast3r_slam/{seq}/K_intrinsics.npy')
    fovx = 2 * np.degrees(np.arctan(0.5 / K[0, 0]))
    return fovx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, required=True, default='09_outdoor_walk')
    args = parser.parse_args()
    seq = args.seq

    fovx_dict = {}

    emdb = [
        '09_outdoor_walk',
        '19_indoor_walk_off_mvs',
        '20_outdoor_walk',
        '24_outdoor_long_walk',
        '27_indoor_walk_off_mvs',
        '28_outdoor_walk_lunges',
        '29_outdoor_stairs_up',
        '30_outdoor_stairs_down',
        '35_indoor_walk',
        '36_outdoor_long_walk',
        '37_outdoor_run_circle',
        '40_indoor_walk_big_circle',
        '48_outdoor_walk_downhill',
        '49_outdoor_big_stairs_down',
        '55_outdoor_walk',
        '56_outdoor_stairs_up_down',
        '57_outdoor_rock_chair',
        '58_outdoor_parcours',
        '61_outdoor_sit_lie_walk',
        '64_outdoor_skateboard',
        '65_outdoor_walk_straight',
        '77_outdoor_stairs_up',
        '78_outdoor_stairs_up_down',
        '79_outdoor_walk_rectangle',
        '80_outdoor_walk_big_circle',
    ]

    for seq in emdb:
        fovx = compute_fov_x(seq)
        print(f"{seq}'s fovx: {fovx}")
        resize_and_copy_keyframes(seq)
        fovx_dict[seq] = fovx

    # 保存到 JSON 文件
    with open('fovx_dict.json', 'w') as f:
        json.dump(fovx_dict, f, indent=4)

