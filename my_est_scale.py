import open3d as o3d
import numpy as np
import cv2
import os
import glob
import argparse
import json


emdb2_names = [
    # "09_outdoor_walk",
    # "19_indoor_walk_off_mvs",
    # "20_outdoor_walk",
    # "24_outdoor_long_walk",
    # "27_indoor_walk_off_mvs",
    # "28_outdoor_walk_lunges",
    "29_outdoor_stairs_up",
    # "30_outdoor_stairs_down",
    # "35_indoor_walk",
    # "36_outdoor_long_walk",
    # "37_outdoor_run_circle",
    # "40_indoor_walk_big_circle",
    # "48_outdoor_walk_downhill",
    # "49_outdoor_big_stairs_down",
    # "55_outdoor_walk",
    # "56_outdoor_stairs_up_down",
    # "57_outdoor_rock_chair",
    # "58_outdoor_parcours",
    # "61_outdoor_sit_lie_walk",
    # "65_outdoor_walk_straight",
    # "64_outdoor_skateboard",
    # "77_outdoor_stairs_up",
    # "78_outdoor_stairs_up_down",
    # "79_outdoor_walk_rectangle",
    # "80_outdoor_walk_big_circle",
]

def umeyama_with_scale(src, tgt):
    assert src.shape == tgt.shape
    n, dim = src.shape

    # 计算均值
    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)

    # 去中心化
    src_demean = src - mu_src
    tgt_demean = tgt - mu_tgt

    # 协方差矩阵
    cov = tgt_demean.T @ src_demean / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    var_src = np.var(src_demean, axis=0).sum()
    scale = np.trace(np.diag(D) @ S) / var_src

    t = mu_tgt - scale * R @ mu_src

    # 构造 4x4 变换矩阵
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return T, scale

def compute_bbox_size(points):
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    return max_pt - min_pt

def process_frame(seq_name, frame_id):
    base_log_path = f"./logs_mast3r_slam/{seq_name}/keyframe_pcd/image"
    pcd_path = os.path.join(base_log_path, f"{frame_id}_canon.ply")
    mask_path = f"./logs_mast3r_slam/{seq_name}/keyframe_pcd/image/{frame_id}_mask.npy"
    smpl_path = f"./results/{seq_name}/smpls/0/{frame_id}_cam.obj"
    moge_path = f"/home/shenwenhao/MoGe/output/{seq_name}/{frame_id}/pointmap.ply"
    moge_human_path = f"/home/shenwenhao/MoGe/output/{seq_name}/{frame_id}/human_pointmap.ply"
    output_dir = f"./vis/{seq_name}"
    os.makedirs(output_dir, exist_ok=True)

    smpl_mesh = o3d.io.read_triangle_mesh(smpl_path)

    # Load point clouds
    pcd = o3d.io.read_point_cloud(pcd_path) 
    pointmap = np.asarray(pcd.points).reshape(512, 384, 3)                  # SLAM cannonical pointmap
    colors = np.asarray(pcd.colors).reshape(512, 384, 3)

    pcd_moge = o3d.io.read_point_cloud(moge_path)
    pointmap_moge = np.asarray(pcd_moge.points).reshape(512, 384, 3)
    colors_moge = np.asarray(pcd_moge.colors).reshape(512, 384, 3)

    pcd_moge_human = o3d.io.read_point_cloud(moge_human_path)
    # labels = np.array(pcd_moge_human.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))
    # inlier_indices = np.where(labels != -1)[0]
    # pcd_moge_human = pcd_moge_human.select_by_index(inlier_indices)


    # Process mask
    mask_bool = np.load(mask_path).reshape(512, 384)
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    mask_bool = mask_eroded > 10                                        # valid points
    # print(mask_bool.sum())
    if mask_bool.sum() < 10000:
        return

    res_points = pointmap[mask_bool]                                    # valid SLAM bkgd points
    res_color = colors[mask_bool]
    res_points_moge = pointmap_moge[mask_bool]                          # valid moge points
    res_color_moge = colors_moge[mask_bool]

    # valid SLAM bkgd points
    pcd_res = o3d.geometry.PointCloud()
    pcd_res.points = o3d.utility.Vector3dVector(res_points)
    pcd_res.colors = o3d.utility.Vector3dVector(res_color)

    # valid MOGE bkgd points
    pcd_res_moge = o3d.geometry.PointCloud()
    pcd_res_moge.points = o3d.utility.Vector3dVector(res_points_moge)
    pcd_res_moge.colors = o3d.utility.Vector3dVector(res_color_moge)

    pcd_res_moge_clean, ind = pcd_res_moge.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    pcd_res_clean = pcd_res.select_by_index(ind)
    pcd_res_moge = pcd_res_moge_clean
    pcd_res = pcd_res_clean


    if False:
        merged_points = np.vstack([res_points, res_points_moge])
        merged_colors = np.vstack([res_color, res_color_moge])
        pcd_merge = o3d.geometry.PointCloud()
        pcd_merge.points = o3d.utility.Vector3dVector(merged_points)
        pcd_merge.colors = o3d.utility.Vector3dVector(merged_colors)
        o3d.io.write_point_cloud(f"{output_dir}/{frame_id}_merged.ply", pcd_merge)

    T, s = umeyama_with_scale(np.asarray(pcd_res_moge.points), np.asarray(pcd_res.points))
    # print(f"[{frame_id}] MOGE -> SLAM estimated scale: {s:.4f}")
    # scale_list.append(s)

    pcd_res_moge.transform(T)
    pcd_moge_human.transform(T)
    pts = np.asarray(pcd_moge_human.points)
    valid_mask = np.isfinite(pts).all(axis=1)
    clean_points = pts[valid_mask]

    # 替换掉原始点
    pcd_moge_human.points = o3d.utility.Vector3dVector(clean_points)
    _, ind_ = pcd_moge_human.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
    pcd_moge_human = pcd_moge_human.select_by_index(ind_)


    # o3d.io.write_point_cloud(f"{output_dir}/{frame_id}_moge_human_aligned.ply", pcd_moge_human)
    # o3d.io.write_triangle_mesh(f"{output_dir}/{frame_id}_smpl.ply", smpl_mesh)
    pcd_smpl = o3d.geometry.PointCloud()
    pcd_smpl.points = o3d.utility.Vector3dVector(np.asarray(smpl_mesh.vertices))
    pcd_smpl.paint_uniform_color([1.0, 0.0, 0.0])
    combined_mesh = pcd_smpl + pcd_moge_human
    o3d.io.write_point_cloud(f"{output_dir}/{frame_id}_moge_human_aligned.ply", combined_mesh)



    pcd_res.paint_uniform_color([0.2, 0.8, 0.2])      # 绿色
    pcd_res_moge.paint_uniform_color([0.8, 0.2, 0.2])  # 红色
    pcd_moge_human.paint_uniform_color([0.4, 0.6, 1.0])  # 蓝色


    combined_points = np.vstack((
        np.asarray(pcd_res.points),
        np.asarray(pcd_res_moge.points),
        np.asarray(pcd_moge_human.points)
    ))
    combined_colors = np.vstack((
        np.asarray(pcd_res.colors),
        np.asarray(pcd_res_moge.colors),
        np.asarray(pcd_moge_human.colors)
    ))
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    o3d.io.write_point_cloud(f"{output_dir}/{frame_id}_aligned_merged.ply", combined_pcd)

    bbox_smpl = compute_bbox_size(np.asarray(smpl_mesh.vertices))
    bbox_moge = compute_bbox_size(np.asarray(pcd_moge_human.points))
    # print(bbox_moge)
    # print(bbox_smpl)
    ratios =  bbox_smpl / bbox_moge
    print(f"[{frame_id}] Bounding box Y-axis ratio (SMPL/SLAM): {ratios[1]:.4f}")
    # ratio_list.append(ratios[1])
    return s, ratios[1]


def main():
    avg_ratio_dict = {}
    
    for seq in emdb2_names:
        print(f"\n=======Processing sequence: {seq}=======")
        
        base_log_path = f"./logs_mast3r_slam/{seq}/keyframe_pcd/image"
        can_2_w_scale_path = f"./logs_mast3r_slam/{seq}/canon_to_w_scale.json"
        with open(can_2_w_scale_path, 'r') as f:
            can_2_w_scale = json.load(f)
        pcd_files = sorted(glob.glob(os.path.join(base_log_path, "*_canon.ply")))

        scale_list = []
        ratio_list = []

        for pcd_file in pcd_files:
            filename = os.path.basename(pcd_file)
            frame_id = filename.split("_")[0]
            try:
                s, ratio = process_frame(seq, frame_id)
                scale_list.append(s)
                can2w = can_2_w_scale[str(int(frame_id))]
                # NOTE: Emperical clip
                # if 0.9 < ratio < 1.4:
                if ratio > 0:
                    ratio = ratio / can2w
                    print(f"[{frame_id}] SMPL/SLAM): {ratio:.4f}")
                    ratio_list.append(ratio)
            except Exception as e:
                print(f"[{frame_id}] Error processing frame: {e}")

        # print("\n=== Summary ===")
        if ratio_list:
            avg_ratio = sum(ratio_list) / len(ratio_list)
            print(f"{seq} Average SMPL/SLAM ratio: {avg_ratio:.4f}")
        else:
            avg_ratio = None
            print(f"{seq} Average SMPL/SLAM ratio WRONG !!!!!!!!!!")

        avg_ratio_dict[seq] = avg_ratio

        stats_output_path = f"./logs_mast3r_slam/{seq}/alignment_stats_.json"
        os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
        with open(stats_output_path, 'w') as f:
            json.dump({
                "scale_list": scale_list,
                "ratio_list": ratio_list,
                "avg_ratio": avg_ratio
            }, f, indent=4)
        print(f"Saved stats to {stats_output_path}")
    
    with open("avg_ratios_new.json", "w") as f:
        json.dump(avg_ratio_dict, f, indent=4)
    
    print(f"Saved stats to avg_ratios.json")




if __name__ == "__main__":
    main()