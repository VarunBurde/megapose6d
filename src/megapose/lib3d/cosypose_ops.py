"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



# Third Party
import torch
import os
from ..ngp_renderer.ngp_render_api import ngp_render
from megapose.config import LOCAL_DATA_DIR
import numpy as np
import json
import cv2

# Local Folder
from .rotations import (
    compute_rotation_matrix_from_ortho6d,
    compute_rotation_matrix_from_quaternions,
)
from .transform_ops import invert_transform_matrices, transform_pts

l1 = lambda diff: diff.abs()
l2 = lambda diff: diff**2


def pose_update_with_reference_point(TCO, K, vxvyvz, dRCO, tCR):
    bsz = len(TCO)
    assert TCO.shape[-2:] == (4, 4)
    assert K.shape[-2:] == (3, 3)
    assert dRCO.shape[-2:] == (3, 3)
    assert vxvyvz.shape[-1] == 3
    assert tCR.shape == (bsz, 3)

    # Translation in image space
    zsrc = tCR[:, [2]]
    vz = vxvyvz[:, [2]]
    ztgt = vz * zsrc

    vxvy = vxvyvz[:, :2]
    fxfy = K[:, [0, 1], [0, 1]]
    xsrcysrc = tCR[:, :2]
    tCR_out = tCR.clone()
    tCR_out[:, 2] = ztgt.flatten()
    tCR_out[:, :2] = ((vxvy / fxfy) + (xsrcysrc / zsrc.repeat(1, 2))) * ztgt.repeat(1, 2)

    tCO_out = dRCO @ (TCO[:, :3, 3] - tCR).unsqueeze(-1) + tCR_out.unsqueeze(-1)
    tCO_out = tCO_out.squeeze(-1)
    TCO_out = TCO.clone().detach()
    TCO_out[:, :3, 3] = tCO_out
    TCO_out[:, :3, :3] = dRCO @ TCO[:, :3, :3]
    return TCO_out


def loss_CO_symmetric(TCO_possible_gt, TCO_pred, points, l1_or_l2=l1):
    bsz = TCO_possible_gt.shape[0]
    assert TCO_possible_gt.shape[0] == bsz
    assert len(TCO_possible_gt.shape) == 4 and TCO_possible_gt.shape[-2:] == (4, 4)
    assert TCO_pred.shape == (bsz, 4, 4)
    assert points.dim() == 3 and points.shape[0] == bsz and points.shape[-1] == 3

    TCO_points_possible_gt = transform_pts(TCO_possible_gt, points)
    TCO_pred_points = transform_pts(TCO_pred, points)
    losses_possible = l1_or_l2(
        (TCO_pred_points.unsqueeze(1) - TCO_points_possible_gt).flatten(-2, -1)
    ).mean(-1)
    loss, min_id = losses_possible.min(dim=1)
    TCO_assign = TCO_possible_gt[torch.arange(bsz), min_id]
    return loss, TCO_assign


def loss_refiner_CO_disentangled_reference_point(
    TCO_possible_gt,
    TCO_input,
    refiner_outputs,
    K_crop,
    points,
    tCR,
):
    # MegaPose
    from megapose.lib3d.transform_ops import invert_transform_matrices

    bsz = TCO_possible_gt.shape[0]
    assert TCO_possible_gt.shape[0] == bsz
    assert TCO_input.shape[0] == bsz
    assert refiner_outputs.shape == (bsz, 9)
    assert K_crop.shape == (bsz, 3, 3)
    assert points.dim() == 3 and points.shape[0] == bsz and points.shape[-1] == 3
    assert TCO_possible_gt.dim() == 4 and TCO_possible_gt.shape[-2:] == (4, 4)
    assert tCR.shape == (bsz, 3)

    dR = compute_rotation_matrix_from_ortho6d(refiner_outputs[:, 0:6])
    vxvy = refiner_outputs[:, 6:8]
    vz = refiner_outputs[:, 8]
    TCO_gt = TCO_possible_gt[:, 0]
    fxfy = K_crop[:, [0, 1], [0, 1]]

    dR_gt = TCO_gt[:, :3, :3] @ TCO_input[:, :3, :3].permute(0, 2, 1)
    tCO_gt = TCO_gt[:, :3, [-1]]
    tCR_out_gt = tCO_gt - dR_gt @ (TCO_input[:, :3, [-1]] - tCR.unsqueeze(-1))
    tCR_out_gt = tCR_out_gt.squeeze(-1)

    vz_gt = tCR_out_gt[:, [2]] / tCR[:, [2]]
    vxvy_gt = fxfy * (tCR_out_gt[:, :2] / tCR_out_gt[:, [2]] - tCR[:, :2] / tCR[:, [2]])

    # First term
    TCO_pred_orn = TCO_gt.clone()
    TCO_pred_orn[:, :3, :3] = pose_update_with_reference_point(
        TCO_input, K_crop, torch.cat((vxvy_gt, vz_gt), dim=-1), dR, tCR
    )[:, :3, :3].to(TCO_pred_orn.dtype)

    # Second term: influence of vxvy
    TCO_pred_xy = TCO_gt.clone()
    TCO_pred_xy[:, :2, [3]] = pose_update_with_reference_point(
        TCO_input, K_crop, torch.cat((vxvy, vz_gt), dim=-1), dR_gt, tCR
    )[:, :2, [3]].to(TCO_pred_xy.dtype)

    # Third term: influence of vz
    TCO_pred_z = TCO_gt.clone()
    TCO_pred_z[:, [2], [3]] = pose_update_with_reference_point(
        TCO_input, K_crop, torch.cat((vxvy_gt, vz.unsqueeze(-1)), dim=-1), dR_gt, tCR
    )[:, [2], [3]].to(TCO_pred_z.dtype)

    loss_orn, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_orn, points, l1_or_l2=l1)
    loss_xy, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_xy, points, l1_or_l2=l1)
    loss_z, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_z, points, l1_or_l2=l1)

    loss = loss_orn + loss_xy + loss_z
    loss_data = {
        "loss_orn": loss_orn,
        "loss_xy": loss_xy,
        "loss_z": loss_z,
        "loss": loss,
    }
    return loss, loss_data


def TCO_init_from_boxes(z_range, boxes, K):
    # Used in the paper
    assert len(z_range) == 2
    assert boxes.shape[-1] == 4
    assert boxes.dim() == 2
    bsz = boxes.shape[0]
    uv_centers = (boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2
    z = (
        torch.as_tensor(z_range)
        .mean()
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(bsz, 1)
        .to(boxes.device)
        .to(boxes.dtype)
    )
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    xy_init = ((uv_centers - cxcy) * z) / fxfy
    TCO = torch.eye(4).unsqueeze(0).to(torch.float).to(boxes.device).repeat(bsz, 1, 1)
    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()
    return TCO


def TCO_init_from_boxes_autodepth_with_R(boxes_2d, model_points_3d, K, R):
    """
    Args:
        boxes_2d: [B,4], in (xmin, ymin, xmax, ymax) convention
        model_points_3d: [B,N,3]
        K: [B,3,3]
        R: [B,3,3]

    Returns:
        TCO: [B,4,4]
    """
    # User in BOP20 challenge
    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    bsz = boxes_2d.shape[0]
    z_guess = 1.0
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    TCO = (
        torch.tensor([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, z_guess], [0, 0, 0, 1]])
        .to(torch.float)
        .to(boxes_2d.device)
        .repeat(bsz, 1, 1)
    )
    TCO[:, :3, :3] = R
    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy
    TCO[:, :2, 3] = xy_init

    C_pts_3d = transform_pts(TCO, model_points_3d)


    if bsz > 0:
        deltax_3d = C_pts_3d[:, :, 0].max(dim=1).values - C_pts_3d[:, :, 0].min(dim=1).values
        deltay_3d = C_pts_3d[:, :, 1].max(dim=1).values - C_pts_3d[:, :, 1].min(dim=1).values
    else:
        deltax_3d = C_pts_3d[:, 0, 0]
        deltay_3d = C_pts_3d[:, 0, 1]


    bb_deltax = (boxes_2d[:, 2] - boxes_2d[:, 0]) + 1
    bb_deltay = (boxes_2d[:, 3] - boxes_2d[:, 1]) + 1

    print("bbox", boxes_2d[0])
    print("bb_xy_centers", bb_xy_centers[0])
    print("bb_deltax", bb_deltax[0])
    print("bb_deltay", bb_deltay[0])
    print("deltax_3d", deltax_3d[0])
    print("deltay_3d", deltay_3d[0])

    z_from_dx = fxfy[:, 0] * deltax_3d / bb_deltax
    z_from_dy = fxfy[:, 1] * deltay_3d / bb_deltay

    z = (z_from_dy.unsqueeze(1) + z_from_dx.unsqueeze(1)) / 2
    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy

    # print("z", z)
    # print("xy_init", xy_init)

    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()
    return TCO


def TCO_init_from_boxes_zup_autodepth(boxes_2d, model_points_3d, K):
    # User in BOP20 challenge
    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    bsz = boxes_2d.shape[0]
    z_guess = 1.0
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    TCO = (
        torch.tensor([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, z_guess], [0, 0, 0, 1]])
        .to(torch.float)
        .to(boxes_2d.device)
        .repeat(bsz, 1, 1)
    )
    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy
    TCO[:, :2, 3] = xy_init

    C_pts_3d = transform_pts(TCO, model_points_3d)

    if bsz > 0:
        deltax_3d = C_pts_3d[:, :, 0].max(dim=1).values - C_pts_3d[:, :, 0].min(dim=1).values
        deltay_3d = C_pts_3d[:, :, 1].max(dim=1).values - C_pts_3d[:, :, 1].min(dim=1).values
    else:
        deltax_3d = C_pts_3d[:, 0, 0]
        deltay_3d = C_pts_3d[:, 0, 1]

    bb_deltax = (boxes_2d[:, 2] - boxes_2d[:, 0]) + 1
    bb_deltay = (boxes_2d[:, 3] - boxes_2d[:, 1]) + 1

    z_from_dx = fxfy[:, 0] * deltax_3d / bb_deltax
    z_from_dy = fxfy[:, 1] * deltay_3d / bb_deltay

    z = (z_from_dy.unsqueeze(1) + z_from_dx.unsqueeze(1)) / 2

    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()
    return TCO


def TCO_init_from_boxes_and_nerf(boxes_2d, K, label, image, R):

    z_guess = 1.0
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    Extrinsics = np.eye(4)
    Extrinsics[2, 3] = z_guess
    resolution = [image.shape[3], image.shape[2]]

    ngp_data_path = os.path.join(LOCAL_DATA_DIR, "examples", label[0], "ngp_weight")

    weight_path = os.path.join(ngp_data_path, "base.ingp")
    world_tranformation = json.loads(open(
        os.path.join(ngp_data_path, "scale.json")).read())
    mesh_transformation = np.array(world_tranformation['transformation'])
    mesh_scale = world_tranformation["scale"]

    ngp_renderer = ngp_render(weight_path)
    ngp_renderer.set_resolution(resolution)
    ngp_renderer.set_exposure(0.0)

    rgb_images = []

    # list of poses of camera to render depth images from each face of cube
    cube_poses = []
    # cube_poses.append(np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, z_guess], [0, 0, 0, 1]]))
    # cube_poses.append(np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, z_guess], [0, 0, 0, 1]]))
    cube_poses.append(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, z_guess], [0, 0, 0, 1]]))
    cube_poses.append(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, z_guess], [0, 0, 0, 1]]))
    cube_poses.append(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, z_guess], [0, 0, 0, 1]]))
    cube_poses.append(np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, z_guess], [0, 0, 0, 1]]))


    # remder depth images from each face of cube
    for i in range(cube_poses.__len__()):
        ngp_renderer.set_camera_matrix(cube_poses[i], mesh_scale, mesh_transformation)
        rgb_image = ngp_renderer.get_image_from_tranform("Shade")
        rgb_images.append(rgb_image)
        # cv2.imwrite(os.path.join(ngp_data_path, "TCO", "image" + str(i) + ".png"), rgb_image)


    # find the max bbox from the rendered images
    corners = []
    for i in range(rgb_images.__len__()):
        depth_image = cv2.GaussianBlur(rgb_images[i], (5, 5), 0)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(depth_image, dtype=np.uint8)
        mask[depth_image > 0] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        corners.append([x, y, (x + w), (y + h)])

    x = min([i[0] for i in corners])
    y = min([i[1] for i in corners])
    w = max([i[2] for i in corners])
    h = max([i[3] for i in corners])

    intrinsics = np.eye(4)
    intrinsics[0, 0] = K[0][0][0]
    intrinsics[1, 1] = K[0][1][1]
    intrinsics[0, 2] = K[0][0][2]
    intrinsics[1, 2] = K[0][1][2]

    # 4 points with pixel coordinates
    points = np.array([[x, y], [w, y], [w, h], [x, h]])
    points = np.concatenate([points, np.ones((4, 2))], axis=1)

    # project the 2d bbox to 3d
    Projection_matrix = intrinsics @ Extrinsics

    # 3d points
    points_3d = np.linalg.inv(Projection_matrix) @ points.T

    points_3d = points_3d.T

    min_x = min(points_3d[:, 0])
    max_x = max(points_3d[:, 0])
    min_y = min(points_3d[:, 1])
    max_y = max(points_3d[:, 1])

    deltax_3d = max_x - min_x
    deltay_3d = max_y - min_y

    bsz = boxes_2d.shape[0]
    TCO = (
        torch.tensor([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, z_guess], [0, 0, 0, 1]])
        .to(torch.float)
        .to(boxes_2d.device)
        .repeat(bsz, 1, 1)
    )
    TCO[:, :3, :3] = R

    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy
    TCO[:, :2, 3] = xy_init

    bb_deltax = (boxes_2d[:, 2] - boxes_2d[:, 0]) + 1
    bb_deltay = (boxes_2d[:, 3] - boxes_2d[:, 1]) + 1

    # deltax_3d = w
    # deltay_3d = h

    print("bbox_nerf", boxes_2d[0])
    print("bb_xy_centers_nerf", bb_xy_centers[0])
    print("bb_deltax_nerf", bb_deltax[0])
    print("bb_deltay_nerf", bb_deltay[0])
    print("deltax_3d_nerf", deltax_3d)
    print("deltay_3d_nerf", deltay_3d)

    z_from_dx = fxfy[:, 0] * deltax_3d / bb_deltax
    z_from_dy = fxfy[:, 1] * deltay_3d / bb_deltay

    z = (z_from_dy.unsqueeze(1) + z_from_dx.unsqueeze(1)) / 2

    # print("z_nerf", z)

    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    # print("xy_init_nerf", xy_init)
    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()

    return TCO


def TCO_init_from_boxes_v3(layer, boxes, K):
    # TODO: Clean these 2 functions
    # MegaPose
    from megapose.math_utils.meshes import get_T_offset

    bsz = len(boxes)
    assert len(K) == bsz
    pts = layer(layer.joints_default.unsqueeze(0)).repeat(bsz, 1, 1)
    T_offset = get_T_offset(pts)
    pts = transform_pts(invert_transform_matrices(T_offset), pts)
    z = (
        torch.as_tensor((1.0, 1.0))
        .mean()
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(bsz, 1)
        .to(boxes.device)
        .to(boxes.dtype)
    )
    TCO = _TCO_init_from_boxes_v2(z, boxes, K)
    pts2d = project_points(pts, K, TCO)
    deltax = pts2d[..., 0].max() - pts2d[..., 0].min()
    deltay = pts2d[..., 1].max() - pts2d[..., 1].min()

    bb_deltax = boxes[:, 2] - boxes[:, 0]
    bb_deltay = boxes[:, 3] - boxes[:, 1]

    ratio_x = deltax / bb_deltax
    ratio_y = deltay / bb_deltay

    z2 = z * (ratio_y.unsqueeze(1) + ratio_x.unsqueeze(1)) / 2
    TCO = _TCO_init_from_boxes_v2(z2, boxes, K)
    return TCO


def init_K_TCO_from_boxes(boxes_2d, model_points_3d, z_guess, resolution):
    # z_guess: float, typically 1.0
    # resolution: input resolution
    device = boxes_2d.device
    H, W = min(resolution), max(resolution)
    bsz = boxes_2d.shape[0]

    z = torch.as_tensor(z_guess).unsqueeze(0).unsqueeze(0).repeat(bsz, 1).to(device).float()
    TCO = torch.eye(4).unsqueeze(0).to(torch.float).to(device).repeat(bsz, 1, 1)
    TCO[:, 2, 3] = z.flatten()

    C_pts_3d = transform_pts(TCO, model_points_3d)
    deltax_3d = C_pts_3d[:, :, 0].max(dim=1).values - C_pts_3d[:, :, 0].min(dim=1).values
    deltay_3d = C_pts_3d[:, :, 1].max(dim=1).values - C_pts_3d[:, :, 1].min(dim=1).values

    bb_deltax = boxes_2d[:, 2] - boxes_2d[:, 0]
    bb_deltay = boxes_2d[:, 3] - boxes_2d[:, 1]

    f_from_dx = bb_deltax * z_guess / deltax_3d
    f_from_dy = bb_deltay * z_guess / deltay_3d
    f = f_from_dy

    K = torch.eye(3, dtype=torch.float, device=device).unsqueeze(0).repeat(bsz, 1, 1)
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K[:, 0, 2] = W / 2
    K[:, 1, 2] = H / 2

    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy
    TCO[:, :2, 3] = xy_init
    return K, TCO


def TCO_init_from_boxes_zup(z_range, boxes, K):
    assert len(z_range) == 2
    assert boxes.shape[-1] == 4
    assert boxes.dim() == 2
    bsz = boxes.shape[0]
    uv_centers = (boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2
    z = (
        torch.as_tensor(z_range)
        .mean()
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(bsz, 1)
        .to(boxes.device)
        .to(boxes.dtype)
    )
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    xy_init = ((uv_centers - cxcy) * z) / fxfy
    TCO = (
        torch.tensor([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        .to(torch.float)
        .to(boxes.device)
        .repeat(bsz, 1, 1)
    )
    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()
    return TCO
