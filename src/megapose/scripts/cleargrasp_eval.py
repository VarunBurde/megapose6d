# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay
from ycb_data_converter.load_files import loader
logger = get_logger(__name__)
import csv
from tqdm import tqdm
import json

def load_observation(
    example_dir: Path,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    rgb = np.array(Image.open(example_dir / "image_rgb.png"), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "image_depth.png"), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(
    example_dir: Path,
) -> DetectionsType:
    input_object_data = load_object_data(example_dir / "inputs/object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


def make_object_dataset(example_dir: Path, label :str) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    label = label
    mesh_path = example_dir / "models" / f"obj_{label}.obj"
    rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

def make_object_dataset_nerf(example_dir: Path, label :str) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    label = label
    mesh_path = example_dir / "model_nerf" / label /f"obj_{label}.obj"
    rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_detections_visualization(
    example_dir: Path,
) -> None:
    rgb, _, _ = load_observation(example_dir, load_depth=False)
    detections = load_detections(example_dir)
    plotter = BokehPlotter()
    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    output_fn = example_dir / "visualizations" / "detections.png"
    output_fn.parent.mkdir(exist_ok=True)
    export_png(fig_det, filename=output_fn)
    logger.info(f"Wrote detections visualization: {output_fn}")
    return


def save_predictions(
    example_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "outputs" / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")
    return

def create_csv(example_dir):

    output_csv_file = example_dir / "cleargrasp_nerf_csv" / "ngp_results.csv"

    json_files_path = example_dir / "cleargrasp_output_nerf"

    row_list = ["scene_id",	"im_id",	"obj_id",	"score",	"R",	"t",	"time"]
    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row_list)
        writer.writeheader()

        for file_name in os.listdir(json_files_path):
            data = None
            json_file_name = os.path.join(json_files_path,file_name)
            with open(json_file_name, "r") as infile:
                data = json.load(infile)

            file_name = str(file_name).strip(".json").split('_')


            scene_id = file_name[0]
            img_id = file_name[1]
            index = file_name[2]
            object_id = file_name[3]
            score = data['score']
            time_ec = data['time']
            T = np.array(data['T']) * 1000
            T = T.tolist()
            T_csv = str(T).strip("[]")
            R = data['R']
            R = np.array(R).reshape(1,9).tolist()
            R_csv = str(R).strip("[]")
            writer.writerow({"scene_id": scene_id,"im_id": img_id,"obj_id": object_id,"score":score,"R":R_csv,"t":T_csv,"time": time_ec})

def run_inference(
    example_dir: Path,
    model_name: str,
) -> None:

    model_info = NAMED_MODELS[model_name]
    path = "/home/testbed/PycharmProjects/megapose6d/local_data/clearGrasp/eval"
    dataset = loader(path)

    scene_id = os.listdir(example_dir / "eval")

    for scene in scene_id:
        bbox_information = example_dir / "eval" / scene / "scene_gt_info.json"
        bbox_information = json.load(open(bbox_information))
        img_id = bbox_information.keys()
        for img in img_id:
            img_data = bbox_information[img]
            for index in range(len(img_data)):
                Rot, TWO, T, K, resolution, bbox, rgb, depth, object_id = dataset.get_gt_RTK_cleargrasp(scene, img,  index)
                observation = ObservationTensor.from_numpy(rgb, depth, K).cuda()

                json_path = example_dir / "cleargrasp_output"
                json_path.mkdir(exist_ok=True)
                file_name = str(scene)  + "_" +  str(img) + "_" + str(index) + "_" + str(object_id) +  ".json"
                json_file_name = os.path.join(json_path,file_name)

                if os.path.exists(json_file_name):
                    continue

                object_data = [{"label": str(object_id).zfill(6), "bbox_modal": bbox}]
                object_data = [ObjectData.from_json(d) for d in object_data]
                detections = make_detections_from_object_data(object_data).cuda()
                # detections = load_detections(example_dir).cuda()
                #
                object_dataset_path = example_dir / "models"
                object_dataset = make_object_dataset(example_dir, str(object_id).zfill(6))
                logger.info(f"Loading model {model_name}.")
                pose_estimator = load_named_model(model_name, object_dataset).cuda()

                logger.info(f"Running inference.")
                output, extra = pose_estimator.run_inference_pipeline(
                    observation, detections=detections, **model_info["inference_parameters"]

                )

                score = output.infos['pose_score'][0]
                labels = output.infos["label"][0]
                poses = output.poses.cpu().numpy()
                poses = poses[0]
                time = extra["time"]
                Rotation = poses[:3,:3]
                Translation = poses[:3, 3]
                data_ycb = {"score": score.tolist(), "labels": labels, "R": Rotation.tolist(), "T": Translation.tolist(), 'time': time}
                # json_path = "/home/testbed/PycharmProjects/megapose6d/local_data/examples/02_cracker_box/ycb_output"

                with open(json_file_name, "w") as outfile:
                    json.dump(data_ycb, outfile, indent=4)


def run_inference_nerf(
    example_dir: Path,
    model_name: str,
) -> None:

    model_info = NAMED_MODELS[model_name]
    path = "/home/testbed/PycharmProjects/megapose6d/local_data/clearGrasp/eval"
    dataset = loader(path)

    scene_id = os.listdir(example_dir / "eval")
    i = 0

    for scene in scene_id:

        bbox_information = example_dir / "eval" / scene / "scene_gt_info.json"
        bbox_information = json.load(open(bbox_information))
        img_id = bbox_information.keys()
        for img in img_id:
            img_data = bbox_information[img]
            for index in range(len(img_data)):
                Rot, TWO, T, K, resolution, bbox, rgb, depth, object_id = dataset.get_gt_RTK_cleargrasp(scene, img,  index)
                observation = ObservationTensor.from_numpy(rgb, depth, K).cuda()
                # if int(object_id) > 7:
                #     continue


                json_path = example_dir / "cleargrasp_output_nerf"
                json_path.mkdir(exist_ok=True)
                file_name = str(scene)  + "_" +  str(img) + "_" + str(index) + "_" + str(object_id) +  ".json"
                json_file_name = os.path.join(json_path,file_name)

                if os.path.exists(json_file_name):
                    continue

                object_data = [{"label": str(object_id).zfill(6), "bbox_modal": bbox}]
                object_data = [ObjectData.from_json(d) for d in object_data]
                detections = make_detections_from_object_data(object_data).cuda()
                # detections = load_detections(example_dir).cuda()
                #
                object_dataset_path = example_dir / "models"
                object_dataset = make_object_dataset_nerf(example_dir, str(object_id).zfill(6))
                logger.info(f"Loading model {model_name}.")
                pose_estimator = load_named_model(model_name, object_dataset).cuda()

                logger.info(f"Running inference.")
                output, extra = pose_estimator.run_inference_pipeline(
                    observation, detections=detections, **model_info["inference_parameters"]

                )

                score = output.infos['pose_score'][0]
                labels = output.infos["label"][0]
                poses = output.poses.cpu().numpy()
                poses = poses[0]
                time = extra["time"]
                Rotation = poses[:3,:3]
                Translation = poses[:3, 3]
                data_ycb = {"score": score.tolist(), "labels": labels, "R": Rotation.tolist(), "T": Translation.tolist(), 'time': time}
                # json_path = "/home/testbed/PycharmProjects/megapose6d/local_data/examples/02_cracker_box/ycb_output"
                breakpoint()
                with open(json_file_name, "w") as outfile:
                    json.dump(data_ycb, outfile, indent=4)


def make_output_visualization(
    example_dir: Path,
) -> None:


    path = "/home/testbed/PycharmProjects/megapose6d/local_data/clearGrasp/cleargrasp_output"
    gt_path = "/home/testbed/PycharmProjects/megapose6d/local_data/clearGrasp/eval"
    dataset = loader(gt_path)
    for file_name in os.listdir(path):


        data = None
        json_file_name = os.path.join(path,file_name)
        with open(json_file_name, "r") as infile:
            data = json.load(infile)


        file_name = str(file_name).strip(".json").split('_')
        scene_id = file_name[0]
        img_id = file_name[1]
        index = file_name[2]
        object_id = file_name[3]
        Rot, TWO, T, K, resolution, bbox, rgb, depth, object_id = dataset.get_gt_RTK_cleargrasp(scene_id, img_id, index)


        # rgb, _, camera_data = load_observation(example_dir, load_depth=False)
        # print(camera_data)

        camera_data = CameraData()
        camera_data.TWC = Transform(np.eye(4))
        camera_data.K = K
        camera_data.resolution = tuple(resolution)

        # print(camera_data)
        # break
        # object_datas = load_object_data(example_dir / "outputs" / "object_data.json")

        # Rtx = list(Rtx.split(','))
        # Rtx = np.array(Rtx).reshape(3,3)
        # Tv = list(Tv.split(','))
        # Tv = np.array(Tv).reshape(3)
        Rtx = data['R']
        Rtx = np.array(Rtx).reshape(3,3)
        Tv = data['T']
        Tv = np.array(Tv)

        TWO_eval = np.eye(4)
        TWO_eval[:3,:3] = Rtx
        TWO_eval[:3,3] = Tv
        # TWO_eval[:3, 3] /= 1000
        Transform_ngp = Transform(TWO_eval)

        object_datas = list()
        # object_data = [{"label": str(object_id).zfill(6), "bbox_modal": bbox}]
        # object_data = [ObjectData.from_json(d) for d in object_data]
        object_datas.append(ObjectData(label=str(object_id).zfill(6), TWO=Transform_ngp))


        object_dataset = make_object_dataset(example_dir, str(object_id).zfill(6))
        renderer = Panda3dSceneRenderer(object_dataset)

        vis_dir = example_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        loc_scene = scene_id + "_" + img_id + "_" + str(index) + "_" + str(object_id)
        result_name = loc_scene + "_NGP" + ".png"
        img_path_exist_check = os.path.join(vis_dir, result_name)

        if os.path.exists(img_path_exist_check):
            print("skipping")
            continue


        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]


        renderings = renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=True,
            render_binary_mask=False,
            render_normals=True,
            copy_arrays=True,
        )[0]

        plotter = BokehPlotter()

        fig_rgb = plotter.plot_image(rgb)
        fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
        contour_overlay = make_contour_overlay(
            rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        fig_contour_overlay = plotter.plot_image(contour_overlay)
        fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)


        # export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
        # export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")

        export_png(fig_all, filename=vis_dir / result_name)
        logger.info(f"Wrote visualizations to {vis_dir}.")


        # ## for gt
        #
        # # Rot, TWO, T, K, resolution, bbox, rgb, depth
        # # Rtx = list(Rtx.split(','))
        # # Rtx = np.array(Rtx).reshape(3, 3)
        # # Tv = list(Tv.split(','))
        # # Tv = np.array(Tv).reshape(3)
        #
        # camera_data_gt = CameraData()
        # camera_data_gt.TWC = Transform(np.eye(4))
        # camera_data_gt.K = K
        # camera_data_gt.resolution = tuple(resolution)
        #
        # TWO_eval_gt = np.eye(4)
        # TWO_eval_gt[:3, :3] = Rot
        # TWO_eval_gt[:3, 3] = T
        # # TWO_eval_gt[:3, 3] /= 1000
        # Transform_gt = Transform(TWO_eval_gt)
        #
        # object_datas_gt = list()
        # object_datas_gt.append(ObjectData(label=object_data_name, TWO=Transform_gt))
        #
        # object_dataset = make_object_dataset(example_dir)
        # renderer = Panda3dSceneRenderer(object_dataset)
        #
        # os.makedirs(example_dir / "visualizations", exist_ok=True)
        #
        # loc_scene = scene_id + "_" + img_id
        # result_name = loc_scene + "_GT" + ".png"
        # img_path_exist_check = os.path.join(vis_dir, result_name)
        #
        # if os.path.exists(img_path_exist_check):
        #     print("skipping")
        #     continue
        #
        # camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data_gt, object_datas_gt)
        # light_datas = [
        #     Panda3dLightData(
        #         light_type="ambient",
        #         color=((1.0, 1.0, 1.0, 1)),
        #     ),
        # ]
        #
        # renderings = renderer.render_scene(
        #     object_datas,
        #     [camera_data],
        #     light_datas,
        #     render_depth=True,
        #     render_binary_mask=False,
        #     render_normals=True,
        #     copy_arrays=True,
        # )[0]
        #
        #
        # plotter = BokehPlotter()
        #
        # fig_rgb = plotter.plot_image(rgb)
        # fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
        # contour_overlay = make_contour_overlay(
        #     rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        # )["img"]
        # fig_contour_overlay = plotter.plot_image(contour_overlay)
        # fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
        #
        # # export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
        # # export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
        #
        # export_png(fig_all, filename=vis_dir / result_name)
        # logger.info(f"Wrote visualizations to {vis_dir}.")
        #
        # i = i + 1
        # if i == 20:
        #     break


def make_output_visualization_nerf(
    example_dir: Path,
) -> None:


    path = "/home/testbed/PycharmProjects/megapose6d/local_data/clearGrasp/cleargrasp_output_nerf"
    gt_path = "/home/testbed/PycharmProjects/megapose6d/local_data/clearGrasp/eval"
    dataset = loader(gt_path)
    for file_name in os.listdir(path):


        data = None
        json_file_name = os.path.join(path,file_name)
        with open(json_file_name, "r") as infile:
            data = json.load(infile)


        file_name = str(file_name).strip(".json").split('_')
        scene_id = file_name[0]
        img_id = file_name[1]
        index = file_name[2]
        object_id = file_name[3]
        Rot, TWO, T, K, resolution, bbox, rgb, depth, object_id = dataset.get_gt_RTK_cleargrasp(scene_id, img_id, index)


        # rgb, _, camera_data = load_observation(example_dir, load_depth=False)
        # print(camera_data)

        camera_data = CameraData()
        camera_data.TWC = Transform(np.eye(4))
        camera_data.K = K
        camera_data.resolution = tuple(resolution)

        # print(camera_data)
        # break
        # object_datas = load_object_data(example_dir / "outputs" / "object_data.json")

        # Rtx = list(Rtx.split(','))
        # Rtx = np.array(Rtx).reshape(3,3)
        # Tv = list(Tv.split(','))
        # Tv = np.array(Tv).reshape(3)
        Rtx = data['R']
        Rtx = np.array(Rtx).reshape(3,3)
        Tv = data['T']
        Tv = np.array(Tv)

        TWO_eval = np.eye(4)
        TWO_eval[:3,:3] = Rtx
        TWO_eval[:3,3] = Tv
        # TWO_eval[:3, 3] /= 1000
        Transform_ngp = Transform(TWO_eval)

        object_datas = list()
        # object_data = [{"label": str(object_id).zfill(6), "bbox_modal": bbox}]
        # object_data = [ObjectData.from_json(d) for d in object_data]
        object_datas.append(ObjectData(label=str(object_id).zfill(6), TWO=Transform_ngp))


        object_dataset = make_object_dataset_nerf(example_dir, str(object_id).zfill(6))
        renderer = Panda3dSceneRenderer(object_dataset)

        vis_dir = example_dir / "visualizations_nerf"
        vis_dir.mkdir(exist_ok=True)

        loc_scene = scene_id + "_" + img_id + "_" + str(index) + "_" + str(object_id)
        result_name = loc_scene + "_NGP" + ".png"
        img_path_exist_check = os.path.join(vis_dir, result_name)

        if os.path.exists(img_path_exist_check):
            print("skipping")
            continue


        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]


        renderings = renderer.render_scene_ngp(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=True,
            render_binary_mask=False,
            render_normals=True,
            copy_arrays=True,
        )[0]

        plotter = BokehPlotter()

        fig_rgb = plotter.plot_image(rgb)
        fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
        contour_overlay = make_contour_overlay(
            rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        fig_contour_overlay = plotter.plot_image(contour_overlay)
        fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)


        # export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
        # export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")

        export_png(fig_all, filename=vis_dir / result_name)
        logger.info(f"Wrote visualizations to {vis_dir}.")



if __name__ == "__main__":
    set_logging_level("info")

    example_dir = LOCAL_DATA_DIR / "clearGrasp"

    run_inference_nerf(example_dir, "megapose-1.0-RGB-multi-hypothesis")

    # create_csv(example_dir)
    # make_output_visualization(example_dir)
    # make_output_visualization_nerf(example_dir)



