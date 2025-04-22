import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        #self.video_nomask = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)
        #self.filterx_nomask = MotionFilter(self.net, self.video_nomask, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        #self.frontend_nomask = DroidFrontend(self.net, self.video_nomask, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            # from visualization import droid_visualization
            from vis_headless import droid_visualization
            print('Using headless ...')
            #/hpc2hdd/home/gzhang292/nanjie/project4/tram/vis
            self.visualizer = Process(target=droid_visualization, args=(self.video, '/hpc2hdd/home/gzhang292/nanjie/project4/tram/vis/video.mp4'))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)
    
    def compute_scale(self, smpl = None, traj=None):
        from vis_headless import compute_scales
        scale = compute_scales(self.video, smpl, traj)

    def visualize_tram(self, save_path = None):
        from vis_headless import save_reconstruction
        #from visualization import droid_visualization
        if save_path != None:
            save_reconstruction(self.video, save_path)
        else:
            save_reconstruction(self.video, '/hpc2hdd/home/gzhang292/nanjie/project4/tram/vis/',ply_name='points.ply')
            #save_reconstruction(self.video_nomask, '/hpc2hdd/home/gzhang292/nanjie/project4/tram/vis/', ply_name='points_nomask.ply')


    def load_weights(self, weights):
        """ load trained model weights """

        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, masked_image, image = None, depth=None, intrinsics=None, mask=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, masked_image, depth, intrinsics, mask)
            # if image != None:
            #     self.filterx_nomask.track(tstamp, image, depth, intrinsics, mask)

            # local bundle adjustment
            self.frontend() # Update

            # if image != None:
            #     self.frontend_nomask()

            # global bundle adjustment
            # self.backend()

    def terminate(self, stream=None, backend=True):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        if backend:
            torch.cuda.empty_cache()
            # print("#" * 32)
            self.backend(7)

            torch.cuda.empty_cache()
            # print("#" * 32)
            self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        
        c2w_scale = (camera_trajectory.inv().data.log()[:, 6].exp())
        return camera_trajectory.inv().data.cpu().numpy(), c2w_scale
    
    def compute_error(self):
        """ compute slam reprojection error """

        del self.frontend

        torch.cuda.empty_cache()
        self.backend(12)

        return self.backend.errors[-1]
        

