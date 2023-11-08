import os
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image

class loader:
    def __init__(self, path_ycb):
        self.ycb_data = path_ycb

    def get_gt_RTK(self,scene_id, img_id, obj_id):
        scene_id = str(scene_id).zfill(6)

        scene_gt_location = os.path.join(self.ycb_data, scene_id, 'scene_gt.json')
        gt_file = json.load(open(scene_gt_location))
        objects =  gt_file[str(img_id)]
        for object in objects:
            if object['obj_id'] == int(obj_id):
                T = np.asarray(object['cam_t_m2c'])*0.001
                Rot = np.asarray(object['cam_R_m2c'])
                Rot = Rot.reshape(3,3)
                r = R.from_matrix(Rot)
                # degree = r.as_euler('zyx', degrees=True)
                TWO = r.as_quat()
                K = self.get_K_mat(scene_id, img_id)
                resolution, image, depth = self.get_resolution_image(scene_id, img_id)
                bbox = self.get_bbox(scene_id, img_id, objects.index(object))

                return Rot, TWO, T, K, resolution, bbox, image, depth

    def get_K_mat(self,scene_id, img_id ):
        scene_id = str(scene_id).zfill(6)
        scene_camera_location = os.path.join(self.ycb_data, scene_id, 'scene_camera.json')
        camera_file = json.load(open(scene_camera_location))
        K = np.asarray(camera_file[str(img_id)]['cam_K'])
        K = K.reshape(3,3)
        return K

    def get_resolution_image(self, scene_id, img_id):
        scene_id = str(scene_id).zfill(6)
        img_id = str(img_id).zfill(6)
        img_name = img_id + '.png'

        img_path = os.path.join(self.ycb_data, scene_id, 'rgb', img_name)
        depth_img_path = os.path.join(self.ycb_data, scene_id, 'depth', img_name)
        # img = cv2.imread(img_path)
        img = np.array(Image.open(img_path), dtype=np.uint8)
        depth = np.array(Image.open(depth_img_path), dtype=np.float32) / 1000
        H, W, C = img.shape
        return [H,C], img, depth

    def get_bbox(self, scene_id, img_id, obj_id):
        scene_id = str(scene_id).zfill(6)
        scene_gt_location = os.path.join(self.ycb_data, scene_id, 'scene_gt_info.json')
        gt_file = json.load(open(scene_gt_location))
        data = gt_file[str(img_id)][obj_id]
        bbox = data["bbox_obj"]
        L, T, W, H = bbox
        bbox = [L, T, L+W, T+H]

        return np.asarray(bbox)

if __name__ == '__main__':
    path = "/home/testbed/PycharmProjects/megapose6d/local_data/ycbv_test_all/test"
    dataset = loader(path)
    Rot, TWO, T, K, resolution, bbox, image, depth = dataset.get_gt_RTK(52, 1, 5)

