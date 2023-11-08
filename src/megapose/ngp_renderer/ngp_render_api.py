import sys
pyngp_path = "/home/varun/PycharmProjects/instant-ngp/build_megapose"

sys.path.append(pyngp_path)
import pyngp as ngp  # noqa
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class ngp_render():
    def __init__(self, weight_path, resolution):
        self.weight_path = weight_path
        self.testbed = ngp.Testbed()
        self.testbed.load_snapshot(weight_path)
        self.screenshot_spp = 1
        self.resolution = resolution
        self.flip_mat = np.array([
                                    [1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]
                                ])

    def load_snapshot(self, snapshot_path):
        self.testbed.load_snapshot(snapshot_path)

    def set_renderer_mode(self, mode):
        if mode == 'Depth':
            self.testbed.render_mode = ngp.RenderMode.Depth
        elif mode == 'Normals':
            self.testbed.render_mode = ngp.RenderMode.Normals
        elif mode == 'Shade':
            self.testbed.render_mode = ngp.RenderMode.Shade


    def set_fov(self, K):
        width = self.resolution[0]
        foclen = K[0, 0]
        fov = np.degrees(2 * np.arctan2(width, 2 * foclen))
        self.testbed.fov = fov
        self.testbed.fov_axis = 0

    def set_exposure(self, exposure):
        self.testbed.exposure = exposure

    def get_image_from_tranform(self, mode):
        self.set_renderer_mode(mode)
        image = self.testbed.render(self.resolution[0], self.resolution[1], self.screenshot_spp, True)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = np.array(image) * 255.0
        return image

    def set_camera_matrix(self, Extrinsics, nerf_scale, mesh_transformation):

        #############################
        # find initial camera matrix
        r = R.from_euler('zyx', [-90,0,-90], degrees=True)
        # W2C = np.eye(4)
        # W2C[:3, :3] = r.as_matrix()
        # # W2C[:3, 3] = np.array([0, 0, 1])
        #
        Extrinsics[:3, 3] /= 350.0
        Extrinsics[:3, 3] *= 1000
        W2C = Extrinsics

        W2C[:3,:3] = np.matmul(W2C[:3,:3], r.as_matrix())

        # convert to C2W
        C2W = np.linalg.inv(W2C)

        # convert camera transformation to openGL coordinate system
        C2W = np.matmul(C2W, self.flip_mat)
        # print("C2W", C2W)

        camera_matrix = C2W[:3, :4]

        self.testbed.set_nerf_camera_matrix(camera_matrix)
