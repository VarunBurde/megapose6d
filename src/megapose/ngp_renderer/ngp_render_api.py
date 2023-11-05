import sys
pyngp_path = "/home/testbed/Projects/instant-ngp/build_megapose"

sys.path.append(pyngp_path)
import pyngp as ngp  # noqa
import cv2
import numpy as np

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

    # def set_fov(self, K):
    #     width = K[0,2] * 2
    #     height = K[1,2] * 2
    #     fl_x = K[0,0]
    #     fl_y = K[1,1]
    #
    #     fov_x = np.arctan2(width / 2, fl_x) * 2 * 180 / np.pi
    #     fov_y = np.arctan2(height / 2, fl_y) * 2 * 180 / np.pi
    #     self.testbed.fov_xy = np.array([fov_x, fov_y])


    def set_fov(self, K):
        width = self.resolution[0]
        foclen = K[0, 0]
        fov = np.degrees(2 * np.arctan2(width, 2 * foclen))
        self.testbed.fov = fov
        self.testbed.fov_axis = 0

    def set_exposure(self, exposure):
        self.testbed.exposure = exposure

    def get_image_from_tranform(self, Extrinsics, mesh_scale, mesh_transformation, mode, exposure = 0.0):

        self.set_renderer_mode(mode)
        camera_matrix = self.get_camera_matrix(Extrinsics,mesh_scale, mesh_transformation)
        self.testbed.set_nerf_camera_matrix(camera_matrix)
        image = self.testbed.render(self.resolution[0], self.resolution[1], self.screenshot_spp, True)
        image = np.array(image) * 255.0
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image

    def get_camera_matrix(self, Extrinsics, mesh_scale, mesh_transformation):

        # convert it to mm
        W2C = Extrinsics
        W2C[:3, 3] = W2C[:3, 3] * mesh_scale

        # Transform the object to the gt mesh
        # W2C = np.matmul(W2C, mesh_transformation)
        C2W = np.linalg.inv(W2C)

        # convert back to meters
        C2W[:3, 3] = C2W[:3, 3] / 100

        camera_matrix = C2W
        camera_matrix = np.matmul(camera_matrix, self.flip_mat)

        return camera_matrix[:-1, :]

    def show_image(self, image):
        cv2.imshow("image", image)
        cv2.waitKey(0)

    def save_image(self, image, path):
        cv2.imwrite(path, image)
