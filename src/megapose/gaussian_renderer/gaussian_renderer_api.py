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
from gaussian_spatting.scene.cameras import Camera
import time
import sys
import cv2


class Gaussian_Renderer_API:
    def __init__(self, path, resolution):
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
        self.resolution = resolution
        self.R = np.eye(3)
        self.T = np.array([0, 0, 0])
        self.flip_mat = np.array([
                                    [1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]
                                ])

    def set_fov(self, K):

        # self.fov_x = np.degrees(2 * np.arctan2(self.resolution[0], 2 * K[0, 0]))
        # self.fov_y = np.degrees(2 * np.arctan2(self.resolution[1], 2 * K[1, 1]))
        self.fovy = 0.8139860440345605
        self.fovx = 0.951948382486254
        # self.screen_center =  np.array([(K[0, 2] / self.resolution[0]), (K[1, 2] / self.resolution[1])])

    def set_camera_matrix(self, Transformation, mesh_scale, mesh_transformation):
        Transformation =  np.linalg.inv(Transformation)
        Transformation = np.matmul(Transformation, self.flip_mat)
        self.R = Transformation[:3, :3]
        self.T = Transformation[:3, 3]

    def get_image_from_tranform(self):
        image = torch.zeros((3, self.resolution[0], self.resolution[1]), dtype=torch.float32, device="cuda")
        view = Camera(colmap_id=None, R=self.R, T=self.T, FoVx=self.fovx, FoVy=self.fovy, image=image, gt_alpha_mask=None,
                        image_name='test', uid=None, data_device="cuda")
        rendering = render(view, self.gaussians, self.pipeline, self.background)["render"]
        rendering = rendering.detach().cpu().numpy()
        rendering = rendering.transpose(2, 1, 0) * 255.0
        return rendering



if __name__ == '__main__':
    resolution = (493, 588)
    Transformation = np.eye(4)
    Transformation[2,3] = 1
    K = np.eye(3)
    K[0,0] = 1000
    K[1,1] = 1200
    K[0,2] = 300
    K[1,2] = 200

    api = Gaussian_Renderer_API("/home/testbed/PycharmProjects/gaussian-splatting/output/d3b2f74a-0", resolution)
    api.set_fov(K)
    api.set_camera_matrix(Transformation, 1, 1)
    render = api.get_image_from_tranform()
    cv2.imwrite("test.png", render)
    breakpoint()


