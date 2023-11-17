#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from gaussian_spatting.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_spatting.gaussian_renderer import render
import torchvision
from gaussian_spatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_spatting.arguments import ModelParams, PipelineParams, get_combined_args, ParamGroup
from gaussian_spatting.gaussian_renderer import GaussianModel
from gaussian_spatting.scene.cameras import MiniCam,Camera
import math
import time
import sys
import cv2
from gaussian_spatting.utils.graphics_utils import focal2fov, fov2focal
from scipy.spatial.transform import Rotation as R

class Gaussian_Renderer_API:
    def __init__(self, path):
        self.model_path = path
        parser = ArgumentParser(description="Testing script parameters")
        sys.argv = [" ", "--model_path", path]
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--skip_train", action="store_true")
        parser.add_argument("--skip_test", action="store_true")
        parser.add_argument("--quiet", action="store_true")

        model = ModelParams(parser, sentinel=True)
        self.pipeline = PipelineParams(parser)
        args = get_combined_args(parser)
        print("Rendering " + args.model_path)

        # Initialize system state (RNG)
        safe_state(args.quiet)

        dataset = model.extract(args)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.gaussians = GaussianModel(dataset.sh_degree)
        self.scene = Scene(dataset, self.gaussians, load_iteration=args.iteration, shuffle=False)
        self.fovx = 0
        self.fovy = 0
        self.resolution = None
        self.view = None
        self.znear = 0.1
        self.zfar = 1000
        self.scale = 1.0
        self.center = None


    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_fov(self, K):
        # self.fovy = focal2fov(K[1,1], self.resolution[1])
        # self.fovx = focal2fov(K[0,0], self.resolution[0])
        self.fovx = 2 * np.arctan2(self.resolution[0], 2 * K[0, 0])
        self.fovy = 2 * np.arctan2(self.resolution[1], 2 * K[1, 1])
        self.center = np.array([(K[0, 2] / self.resolution[0]),(K[1, 2] / self.resolution[1])])
        # print("center", self.center)
        # print("width: " + str(self.resolution[0]), "height: " + str(self.resolution[1]))
        # print("Fovx: " + str(self.fovx), "Fovy: " + str(self.fovy))        # self.fovy = focal2fov(K[1,1], self.resolution[1])
        # self.fovx = focal2fov(K[0,0], self.resolution[0])

    def getWorld2View2(self,R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)

    def getProjectionMatrix(self, znear, zfar, fovX, fovY):

        tanHalfFovY = math.tan(((fovY) / 2))
        tanHalfFovX = math.tan(((fovX) / 2))

        # print(f"fovX: {fovX}, fovY: {fovY}")
        # print("tan fov y",  tanHalfFovY, tanHalfFovX , "tan fov x")


        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        # print("top", top, "bottom", bottom, "right", right, "left", left)
        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left) # smthg/2*right
        P[1, 1] = 2.0 * znear / (top - bottom)
        # P[0, 2] = (right + left) / (right - left)
        # P[1, 2] = (top + bottom) / (top - bottom)
        P[0, 2] = self.center[0] * 2 - 1
        P[1, 2] = self.center[1] * 2 - 1
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        # print(P)
        return P

    def set_camera_matrix(self, Extrinsics, mesh_scale, mesh_transformation):

        # convert the scale to mm to apply the transformation
        Extrinsics[:3, 3] *= 1000

        # apply the alignment transformation
        Extrinsics = np.matmul(Extrinsics, mesh_transformation)

        # convert back to m scale
        Extrinsics[:3,3] /=1000


        # x_trans = Extrinsics[0,3]
        # y_trans = Extrinsics[1,3]
        # Extrinsics[0,3] = y_trans
        # Extrinsics[1,3] = x_trans

        # r = R.from_euler('zyx', [0,0,-90], degrees=True)
        # Extrinsics[:3,:3] = np.matmul(Extrinsics[:3,:3], r.as_matrix())

        # print(R.from_matrix(Extrinsics[:3,:3]).as_euler('zyx', degrees=True))
        # print(Extrinsics[:3,3])

        Rot = np.transpose(Extrinsics[:3,:3])
        Tra = Extrinsics[:3, 3]

        world_view = torch.tensor(self.getWorld2View2(Rot, Tra, scale=self.scale)).transpose(0, 1).cuda()

        projection_matrix = torch.tensor(self.getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy)).transpose(0,1).cuda()
        full_projection = (world_view.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

        self.view = MiniCam( width=self.resolution[0], height=self.resolution[1], fovy=self.fovy, fovx=self.fovx,
                             znear=self.znear, zfar=self.zfar, world_view_transform=world_view, full_proj_transform=full_projection)

        # image = torch.zeros((1, self.resolution[1], self.resolution[0]), dtype=torch.float32, device="cuda")
        # self.view = Camera(colmap_id=0, R=Rot, T=Tra, FoVx=self.fovx, FoVy=self.fovy, image=image, gt_alpha_mask=None,
        #                    image_name="test",uid=0)


    def get_image_from_tranform(self):
        # print(self.view.image_width, self.view.image_height)
        rendering = render(self.view, self.gaussians, self.pipeline, self.background)["render"]
        rendering = rendering.detach().cpu().numpy()
        rendering = rendering.transpose(1, 2, 0) * 255.0
        return rendering


# if __name__ == '__main__':
#     resolution = (493, 588)
#     Transformation = np.eye(4)
#     Transformation[2,3] = 1
#     K = np.eye(3)
#     K[0,0] = 1000
#     K[1,1] = 1200
#     K[0,2] = 300
#     K[1,2] = 200
#
#     api = Gaussian_Renderer_API("/home/testbed/PycharmProjects/gaussian-splatting/output/d3b2f74a-0")
#     api.set_fov(K)
#     api.set_camera_matrix(Transformation, 1, 1)
#     render = api.get_image_from_tranform()
#     cv2.imwrite("test.png", render)
#     breakpoint()


