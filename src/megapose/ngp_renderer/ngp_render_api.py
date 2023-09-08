import sys
import os

pyngp_path = "/home/varun/PycharmProjects/instant-ngp/build_megapose"

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

    def load_snapshot(self, snapshot_path):
        self.testbed.load_snapshot(snapshot_path)

    def set_renderer_mode(self, mode):
        if mode == 'Depth':
            self.testbed.render_mode = ngp.RenderMode.Depth
        elif mode == 'Normals':
            self.testbed.render_mode = ngp.RenderMode.Normals
        elif mode == 'Shade':
            self.testbed.render_mode = ngp.RenderMode.Shade

    def set_camera_matrix(self, K):
        self.testbed.set_nerf_camera_matrix(K)

    def get_image_from_tranform(self, matrix, mode):
        self.set_renderer_mode(mode)
        projection_matrix = self.get_projection_matrix(matrix)
        self.testbed.set_nerf_camera_matrix(projection_matrix)
        image = self.testbed.render(self.resolution[0], self.resolution[1], self.screenshot_spp, True)
        image = np.array(image) * 255.0
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return image

    def get_projection_matrix(self, transform):
        return transform[:-1, :]

    def show_image(self, image):
        cv2.imshow("image", image)
        cv2.waitKey(0)

    def save_image(self, image, path):
        cv2.imwrite(path, image)
