import os
import argparse
import open3d as o3d
import json
import numpy as np


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],  # frustum origin
                               [-half_w, -half_h, frustum_length],  # top-left image corner
                               [half_w, -half_h, frustum_length],  # top-right image corner
                               [half_w, half_h, frustum_length],  # bottom-right image corner
                               [-half_w, half_h, frustum_length]])  # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i + 1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)

    C2W[:3, 3] *= 350

    # convert TCO transformation to mm scale
    # C2W[:3, 3] *= 1000

    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    C2W = np.matmul(C2W, flip_mat)

    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N * 5, 3))  # 5 vertices per frustum
    merged_lines = np.zeros((N * 8, 2))  # 8 lines per frustum
    merged_colors = np.zeros((N * 8, 3))  # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i * 5:(i + 1) * 5, :] = frustum_points
        merged_lines[i * 8:(i + 1) * 8, :] = frustum_lines + i * 5
        merged_colors[i * 8:(i + 1) * 8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def visualize_camera_transform():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=500, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0., 0., 0.])
    things_to_draw = []


    frustums = []

    img_size = [640, 480]
    K = np.array([[570.5,    0,   296.25],
                  [  0,   572,   244.75],
                  [  0,     0,     1.  ]])

    W2C = np.array([[-0.02561679, -0.47549924, -0.87934308, -0.03500213],
                     [ 0.99329297,  0.08710515, -0.07603791, -0.0048417 ],
                     [ 0.11275128, -0.87539315,  0.4700787,   0.25060049],
                     [ 0,          0,          0,          1.        ]])

    W2C[:3, 3] *= 1000

    # W2C[:3, 3] /= 350

    # transform = np.array([[1, 0, 0, 0],
    #                         [0, 0, -1, 0],
    #                         [0, -1, 0, 0],
    #                         [0, 0, 0, 1]])
    # W2C = np.matmul(W2C, transform)

    frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=50, color=[0, 1, 0]))
    cameras = frustums2lineset(frustums)

    geometry_file = "/home/varun/PycharmProjects/megapose6d_varun/local_data/examples/02_cracker_box/base.ply"
    geometry = o3d.io.read_triangle_mesh(geometry_file, True)

    # things to draw
    things_to_draw.append(geometry)
    things_to_draw.append(cameras)
    things_to_draw.append(coord_frame)
    things_to_draw.append(sphere)


    o3d.visualization.draw_geometries(things_to_draw)


if __name__ == '__main__':
    visualize_camera_transform()
