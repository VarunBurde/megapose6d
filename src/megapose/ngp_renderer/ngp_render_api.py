import sys
pyngp_path = "/home/testbed/Projects/instant-ngp/build_megapose"

sys.path.append(pyngp_path)
import pyngp as ngp  # noqa
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class ngp_render():
    def __init__(self, weight_path):
        self.weight_path = weight_path
        self.testbed = ngp.Testbed()
        self.testbed.load_snapshot(weight_path)
        self.screenshot_spp = 1
        self.resolution = None
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

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_fov(self, K):

        # width = self.resolution[0]
        # foclen = K[0, 0]
        # fov = np.degrees(2 * np.arctan2(width, 2 * foclen))
        # self.testbed.fov_axis = 0
        # self.testbed.fov = fov

        # fov_x = np.arctan2(self.resolution[0]/2, K[0,0]) * 2 * 180 / np.pi
        # fov_y = np.arctan2(self.resolution[1]/2, K[1,1]) * 2 * 180 / np.pi

        fov_x = np.degrees(2 * np.arctan2(self.resolution[0], 2 * K[0,0]))
        fov_y = np.degrees(2 * np.arctan2(self.resolution[1], 2 * K[1,1]))
        self.testbed.screen_center = np.array([1 - (K[0,2]/self.resolution[0]), 1 - (K[1,2] /self.resolution[1])])

        self.testbed.fov_axis = 0
        self.testbed.fov = fov_x

        # self.testbed.fov_axis = 1
        # self.testbed.fov = fov_y

        # self.testbed.fov_xy = np.array([fov_x, fov_y])


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
        # convert the scale to mm to apply the transformation
        Extrinsics[:3, 3] *= 1000

        # apply the alignment transformation
        Extrinsics = np.matmul(Extrinsics, mesh_transformation)

        # inital pose of renderer
        r = R.from_euler('zyx', [-90,0,-90], degrees=True)
        Extrinsics[:3,:3] = np.matmul(Extrinsics[:3,:3], r.as_matrix())

        # convert back to m scale
        Extrinsics[:3,3] /=1000

        # convert to nerf scale
        Extrinsics[:3, 3] /= nerf_scale
        Extrinsics[:3, 3] *= 1000

        # convert to C2W
        C2W = np.linalg.inv(Extrinsics)

        # convert camera transformation to openGL coordinate system
        C2W = np.matmul(C2W, self.flip_mat)

        camera_matrix = C2W[:3, :4]

        self.testbed.set_nerf_camera_matrix(camera_matrix)
