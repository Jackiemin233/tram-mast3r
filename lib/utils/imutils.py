"""
This file contains functions that are used to perform data augmentation.
"""
import torch
import numpy as np
from skimage.transform import rotate, resize
import cv2
from torchvision.transforms import Normalize, ToTensor, Compose

import shutil
import os

from lib.core import constants

from tqdm import tqdm

def write_main_mask(input_folder, output_folder):

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

def get_normalization():
    normalize_img = Compose([ToTensor(),
                            Normalize(mean=constants.IMG_NORM_MEAN, 
                                      std=constants.IMG_NORM_STD)
                            ])
    return normalize_img

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale + 1e-6
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0, asint=True):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)

    if asint:
        return new_pt[:2].astype(int)+1
    else:
        return new_pt[:2]+1

def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)
    

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = resize(new_img, res)
    return new_img

def crop_crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)
    

    if new_img.shape[0] > img.shape[0]:
        p = (new_img.shape[0] - img.shape[0]) / 2
        p = int(p)
        new_img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REPLICATE)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = resize(new_img, res)
    return new_img

def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,res[1]+1], center, scale, res, invert=1))-1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    img = resize(img, crop_shape, interp='nearest')
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]
    return new_img

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa

def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def flip_kp(kp):
    """Flip keypoints."""
    if len(kp) == 24:
        flipped_parts = constants.J24_FLIP_PERM
    elif len(kp) == 49:
        flipped_parts = constants.J49_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:,0] = - kp[:,0]
    return kp

def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose


def crop_img(img, center, scale, res, val=255):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.ones(new_shape) * val

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]
    new_img = resize(new_img, res)
    return new_img


def boxes_2_cs(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w, h = x2-x1, y2-y1
    cx, cy = x1+w/2, y1+h/2
    size = np.stack([w, h]).max(axis=0)
    
    centers = np.stack([cx, cy], axis=1)
    scales = size / 200
    return centers, scales


def box_2_cs(box):
    x1,y1,x2,y2 = box[:4].int().tolist()

    w, h = x2-x1, y2-y1
    cx, cy = x1+w/2, y1+h/2
    size = max(w, h)

    center = [cx, cy]
    scale = size / 200
    return center, scale


def est_intrinsics(img_shape):
    h, w, c = img_shape
    img_center = torch.tensor([w/2., h/2.]).float()
    img_focal = torch.tensor(np.sqrt(h**2 + w**2)).float()
    return img_center, img_focal

def copy_images(source_dir, target_dir):
    """
    将源目录中的所有图片文件复制到目标目录，自动处理重名文件
    :param source_dir: 源目录路径
    :param target_dir: 目标目录路径
    """
    # 支持的图片扩展名（可根据需要添加）
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    
    # 确保目标目录存在
    if os.path.exists(target_dir):
        return
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源目录及其所有子目录
    for root, _, files in os.walk(source_dir):
        for filename in files:
            # 检查文件扩展名是否为图片格式
            if filename.lower().endswith(image_extensions):
                src_path = os.path.join(root, filename)
                
                # 构造目标路径并处理重名
                base_name, ext = os.path.splitext(filename)
                dest_path = os.path.join(target_dir, filename)
                duplicate_count = 0
                
                # 如果文件已存在，添加数字后缀
                while os.path.exists(dest_path):
                    duplicate_count += 1
                    new_name = f"{base_name}_{duplicate_count}{ext}"
                    dest_path = os.path.join(target_dir, new_name)
                
                # 复制文件并保留元数据
                shutil.copy2(src_path, dest_path)