import sys
pyngp_path = "/home/testbed/Projects/instant-ngp/build_megapose"

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

        ## debug
        r = R.from_matrix(Extrinsics[:3,:3])
        rotation = r.as_euler('zyx', degrees=True)
        print(rotation, Extrinsics[:3, 3])

        # rotation = R.from_euler("zyx",[180,-140,0], degrees=True)
        rotation = R.from_euler("yzx",[160,180,0], degrees=True)
        W2C = np.eye(4)
        W2C[:3,:3] = rotation.as_matrix()
        W2C[2, 3] = 0.4

        # convert to C2W
        C2W = np.linalg.inv(W2C)

        r = R.from_matrix(C2W[:3,:3])
        rotation = r.as_euler('zyx', degrees=True)
        print(rotation)


        C2W[:3, 3] /= nerf_scale

        # convert TCO transformation to mm scale
        C2W[:3,3] *= 1000

        # C2W[0:3, 1:3] *= -1
        # c2w = C2W[np.array([1, 0, 2, 3]), :]
        # c2w[2, :] *= -1

        C2W = np.matmul(C2W, self.flip_mat)

        camera_matrix= C2W[:-1, :]

        self.testbed.set_nerf_camera_matrix(camera_matrix)

