import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np

# import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import open3d as o3d

# o3d.visualization.webrtc_server.enable_webrtc()

from lietorch import SE3
import geom.projective_ops as pops
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


CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def droid_visualization(video, save_path, device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(0)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0
    print("headless droid_visualization")


    droid_visualization.filter_thresh = 0.3  #0.005

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True   
    
    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        with torch.no_grad():

            with video.get_lock():
                t = video.counter.value 
                dirty_index, = torch.where(video.dirty.clone())
                dirty_index = dirty_index

            if len(dirty_index) == 0:
                return

            video.dirty[dirty_index] = False

            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, 0, dirty_index)
            disps = torch.index_select(video.disps, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(video.images, 0, dirty_index)
            images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
            points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

            thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
            
            count = droid_backends.depth_filter(
                video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)
            
            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))     
        
            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()

                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    del droid_visualization.cameras[ix]

                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    del droid_visualization.points[ix]

                ### add camera actor ###
                cam_actor = create_camera_actor(True)
                cam_actor.transform(pose)
                vis.add_geometry(cam_actor)
                droid_visualization.cameras[ix] = cam_actor
                
                
                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                
                ## add point actor ###
                point_actor = create_point_actor(pts, clr)
                vis.add_geometry(point_actor)
                droid_visualization.points[ix] = point_actor

            ### Hack to save Point Cloud Data and Camnera results ###
            
            # Save points
            pcd_points = o3d.geometry.PointCloud()
            for p in droid_visualization.points.items():
                pcd_points += p[1]
            o3d.io.write_point_cloud(f"{save_path}/points.ply", pcd_points, write_ascii=False)
                
            # Save pose
            pcd_camera = create_camera_actor(True)
            for c in droid_visualization.cameras.items():
                pcd_camera += c[1]

            o3d.io.write_line_set(f"{save_path}/camera.ply", pcd_camera, write_ascii=False)

            ### end ###
            
            # hack to allow interacting with vizualization during inference
            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            droid_visualization.ix += 1
            vis.poll_events()
            vis.update_renderer()

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)

    vis.create_window(height=540, width=960)
    # vis.create_window(height=512, width=384)
    vis.get_render_option().load_from_json("thirdparty/DROID-SLAM//misc/renderoption.json")

    vis.run()
    vis.destroy_window()


def save_reconstruction(video, save_path, ply_name='point.ply', device="cuda:0"):
    """ Saves DROID-SLAM reconstruction results to .PLY files without visualization """
    
    torch.cuda.set_device(device)

    with torch.no_grad():
        with video.get_lock():
            dirty_index, = torch.where(video.dirty.clone())
        #cv2.imwrite('/hpc2hdd/home/gzhang292/nanjie/project4/tram/vis/imagedroid.png', video.images[0].permute(1,2,0).cpu().numpy() * 255)
        if len(dirty_index) == 0:
            print("No new frames to process.")
            return

        # Convert poses to 4x4 matrices
        poses = torch.index_select(video.poses, 0, dirty_index)
        disps = torch.index_select(video.disps, 0, dirty_index)
        Ps = SE3(poses).inv().matrix().cpu().numpy()

        images = torch.index_select(video.images, 0, dirty_index)
        #images = images.cpu()[:, [2, 1, 0], :, :].permute(0, 2, 3, 1) / 255.0
        images = images.cpu()[:, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
        
        #cv2.imwrite('/hpc2hdd/home/gzhang292/nanjie/project4/tram/vis/images?.png', images[20].cpu().permute(1,2,0).numpy())

        points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

        # Depth filtering
        thresh = 0.8 * torch.ones_like(disps.mean(dim=[1, 2]))
        count = droid_backends.depth_filter(video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

        count = count.cpu()
        disps = disps.cpu()
        masks = ((count >= 2) & (disps > 0.3 * disps.mean(dim=[1, 2], keepdim=True)))

        all_pts = []
        all_clr = []

        for i in range(len(dirty_index)): # for each key frames, get the points
            mask = masks[i].reshape(-1)
            pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
            clr = images[i].reshape(-1, 3)[mask].cpu().numpy()

            all_pts.append(pts)
            all_clr.append(clr)

        all_pts = np.concatenate(all_pts, axis=0)
        all_clr = np.concatenate(all_clr, axis=0)

        # Save points to .ply 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.colors = o3d.utility.Vector3dVector(all_clr)
        o3d.io.write_point_cloud(f"{save_path}/{ply_name}", pcd, write_ascii=False)

        print(f"Saved {len(all_pts)} points to {ply_name}")

        # Save camera poses to .ply
        cameras = []
        for pose in Ps:
            cam_actor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            cam_actor.transform(pose)
            cameras.append(cam_actor)

        combined_camera_mesh = sum(cameras, o3d.geometry.TriangleMesh())
        o3d.io.write_triangle_mesh(f"{save_path}/cameras.ply", combined_camera_mesh, write_ascii=False)

        print(f"Saved {len(Ps)} camera poses to cameras.ply")

def compute_scales(video, smpl, traj_ori, device="cuda:0"):

    torch.cuda.set_device(device)
    with torch.no_grad():
        with video.get_lock():
            dirty_index, = torch.where(video.dirty.clone())

        if len(dirty_index) == 0:
            print("No new frames to process.")
            return

        # Convert poses to 4x4 matrices
        poses = torch.index_select(video.poses, 0, dirty_index)
        disps = torch.index_select(video.disps, 0, dirty_index)

        images = torch.index_select(video.images, 0, dirty_index)
        images = images.cpu()[:, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0

        #cv2.imwrite('/hpc2hdd/home/gzhang292/nanjie/project4/tram/vis/images_idx.png', video.images.permute(1,2,0).cpu().numpy())
        #============================================================================================================
        #The tstamp record the frame indexes of key frames - NJ.
        tstamp = torch.index_select(video.tstamp, 0 ,dirty_index)
                
        # Find the longest SMPL sequence (multi-person)
        smpl_l = max(smpl, key = lambda x: len(x['pred_verts']) if 'pred_verts' in x else 0)

        # select SMPL attributes in each key frame
        for k in smpl_l.keys():
            if k == 'smpl_faces':
                continue
            smpl_l[k] = torch.index_select(smpl_l[k], 0, tstamp.cpu().long()) 
            
        traj =  torch.index_select(torch.tensor(traj_ori).cuda(), 0, tstamp.long())
        smpl_l['pred_verts_cam'] = smpl_l['pred_verts'] + smpl_l['pred_trans'] # 2D (SMPL model forward)
        
        pred_cam_t = traj[:, :3].cpu()
        pred_cam_q = traj[:, 3:].cpu()
        pred_cam_r = quaternion_to_matrix(pred_cam_q[:,[3,0,1,2]])

        pred_vert_w = torch.einsum('bij,bnj->bni', pred_cam_r, smpl_l['pred_verts_cam']) + pred_cam_t[:,None] 

        # Save the mesh for visualization - debug
        for idx in range(smpl_l['pred_verts_cam'].shape[0]):
            trimesh.Trimesh(pred_vert_w[idx], smpl[0]['smpl_faces']).export(f'/hpc2hdd/home/gzhang292/nanjie/project4/tram/vis_smpl/smpl_{idx}.obj')

        #============================================================================================================
        points = droid_backends.iproj(SE3(traj).data, disps, video.intrinsics[0]).cpu()

        # Depth filtering
        thresh = 0.8 * torch.ones_like(disps.mean(dim=[1, 2]))
        
        count = droid_backends.depth_filter(video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

        count = count.cpu()
        disps = disps.cpu()
        masks = ((count >= 2) & (disps > 0.2 * disps.mean(dim=[1, 2], keepdim=True)))

        all_pts = []
        all_clr = []

        for i in range(len(dirty_index)): # for each key frames, get the points
            mask = masks[i].reshape(-1)
            pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
            clr = images[i].reshape(-1, 3)[mask].cpu().numpy()

            all_pts.append(pts)
            all_clr.append(clr)

            #NOTE: Debug - NJ - visulize pcs for each frame 
            all_pts_debug = []
            all_clr_debug = []
            all_pts_debug.append(pts)
            all_clr_debug.append(clr)
            pcd = o3d.geometry.PointCloud()
            all_pts_debug = np.concatenate(all_pts_debug, axis=0)
            all_clr_debug = np.concatenate(all_clr_debug, axis=0)
            pcd.points = o3d.utility.Vector3dVector(all_pts_debug) # 
            pcd.colors = o3d.utility.Vector3dVector(all_clr_debug) # Color 
            o3d.io.write_point_cloud(f"/hpc2hdd/home/gzhang292/nanjie/project4/tram/vis_ply/{i}_ply.ply", pcd, write_ascii=False)
            
        all_pts = np.concatenate(all_pts, axis=0)
        all_clr = np.concatenate(all_clr, axis=0)

        # TODO:  WIP 4.1
        return None
    
    
        #Ps = SE3(poses).inv().matrix().cpu().numpy() # [n,4,4] Camera extrisics
        # R = Ps[:,:3, :3]
        # T = Ps[:,:3,-1:]