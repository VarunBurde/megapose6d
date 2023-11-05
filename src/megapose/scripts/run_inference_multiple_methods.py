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
from tqdm import tqdm

# MegaPose
# from megapose.config import LOCAL_DATA_DIR
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

logger = get_logger(__name__)


def load_observation(
    example_dir: Path, img_file: Path, load_depth: bool = False
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    rgb = np.array(Image.open(img_file), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "image_depth.png"), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path, file_path: Path, load_depth: bool = False
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, file_path, load_depth=False)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(
    example_dir: Path,
    img_file: str,
) -> DetectionsType:
    input_object_data = load_object_data(example_dir / img_file)
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


# def make_object_dataset(example_dir: Path, method: str) -> RigidObjectDataset:
#     rigid_objects = []
#     mesh_units = "mm"
#     object_dirs = (example_dir / "meshes" / method).iterdir()
#     # print((example_dir / "meshes" / method))
#     for object_dir in object_dirs:
#         label = object_dir.name
#         mesh_path = None
#         for fn in object_dir.glob("*"):
#             if fn.suffix in {".obj", ".ply"}:
#                 assert not mesh_path, f"there multiple meshes in the {label} directory"
#                 mesh_path = fn
#         assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
#         rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
#         # TODO: fix mesh units
#     rigid_object_dataset = RigidObjectDataset(rigid_objects)
#     return rigid_object_dataset


def make_object_dataset(example_dir: Path, method: str) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes" / method).iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units

    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_detections_visualization(
    example_dir: Path,
) -> None:
    img_dir = os.path.join(example_dir, "images")
    for file in tqdm(sorted(os.listdir(img_dir))):
        file_path = os.path.join(img_dir, file)
        file = file.split(".")[0]
        rgb, _, _ = load_observation(example_dir, file_path, load_depth=False)
        label_file = os.path.join("inputs", file + ".json")
        detections = load_detections(example_dir, label_file)
        plotter = BokehPlotter()
        fig_rgb = plotter.plot_image(rgb)
        fig_det = plotter.plot_detections(fig_rgb, detections=detections)
        output_fn = example_dir / "visualizations" / "detections" / "{}.png".format(file)
        output_fn.parent.mkdir(exist_ok=True)
        export_png(fig_det, filename=output_fn)
        logger.info(f"Wrote detections visualization: {output_fn}")
    return


def save_predictions(
    example_dir: Path, pose_estimates: PoseEstimatesType, file_path: str, method: str
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "outputs" / method / "{}.json".format(file_path)
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    # logger.info(f"Wrote predictions: {output_fn}")
    return


def run_inference(
    example_dir: Path,
    model_name: str,
    method: str,
) -> None:
    model_info = NAMED_MODELS[model_name]
    img_dir = os.path.join(example_dir, "images")
    i = 0
    for file in tqdm(sorted(os.listdir(img_dir))):
        file_path = os.path.join(img_dir, file)
        file = file.split(".")[0]
        output_path = os.path.join(example_dir, "outputs", method, file + ".json")
        if os.path.exists(output_path):
            print("image {} already processed".format(file))
            continue
        observation = load_observation_tensor(
            example_dir, file_path, load_depth=model_info["requires_depth"]
        ).cuda()
        label_file = os.path.join("inputs", file + ".json")
        detections = load_detections(example_dir, label_file)
        object_dataset = make_object_dataset(example_dir, method)

        logger.info(f"Loading model {model_name}.")
        pose_estimator = load_named_model(model_name, object_dataset).cuda()

        logger.info(f"Running inference.")
        ## BUG: IT BREAKS HERE FOR SOME REASON in the inference pipeline
        # Only with second method
        output, _ = pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"]
        )
        save_predictions(example_dir, output, file, method)

    return


def run_inference_new(
    example_dir: Path,
    model_name: str,
    method: str,
    frames2process: int = None,
) -> None:
    """Run inferance on a single example

    Args:
        example_dir (Path): Path to the example directory
        model_name (str): Name of the model to use for inference
        method (str): Name of the method used to recreate the mesh used for inference
        frames2process (int, optional): Number of frames to reconstruct. Defaults to None.
    """
    model_info = NAMED_MODELS[model_name]
    img_dir = os.path.join(example_dir, "images")

    logger.info(f"Loading model {model_name}.")
    for i, file in enumerate(tqdm(sorted(os.listdir(img_dir)))):
        file_path = os.path.join(img_dir, file)
        file = file.split(".")[0]
        output_path = os.path.join(example_dir, "outputs", method, file + ".json")

        if frames2process is not None and i >= frames2process:
            print(f"[INFO]: Stopping after {i} frames.")
            break
        if os.path.exists(output_path):
            print("[INFO]: Image {} already processed".format(file))
            continue

        observation = load_observation_tensor(
            example_dir, file_path, load_depth=model_info["requires_depth"]
        ).cuda()
        label_file = os.path.join("inputs", file + ".json")
        detections = load_detections(example_dir, label_file)
        object_dataset = make_object_dataset(example_dir, method)

        # logger.info(f"Loading model {model_name}.")
        pose_estimator = load_named_model(model_name, object_dataset).cuda()

        # logger.info(f"Running inference.")
        ## BUG: IT BREAKS HERE FOR SOME REASON in the inference pipeline
        # Only with second method
        output, _ = pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"]
        )
        save_predictions(example_dir, output, file, method)
    return

def run_inference_newest(
    example_dir: Path,
    model_name: str,
    method: str,
    frames2process: int = None,
) -> None:
    """Run inferance on a single example

    Args:
        example_dir (Path): Path to the example directory
        model_name (str): Name of the model to use for inference
        method (str): Name of the method used to recreate the mesh used for inference
        frames2process (int, optional): Number of frames to reconstruct. Defaults to None.
    """
    model_info = NAMED_MODELS[model_name]
    img_dir = os.path.join(example_dir, "images")

    logger.info(f"Loading model {model_name}.")
    object_dataset = make_object_dataset(example_dir, method)
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    for i, file in enumerate(tqdm(sorted(os.listdir(img_dir)))):
        file_path = os.path.join(img_dir, file)
        file = file.split(".")[0]
        output_path = os.path.join(example_dir, "outputs", method, file + ".json")

        if frames2process is not None and i >= frames2process:
            logger.info(f"Stopping after {i} frames.")
            break

        if os.path.exists(output_path):
            logger.info(f"Image {file} already processed")
            continue

        observation = load_observation_tensor(
            example_dir, file_path, load_depth=model_info["requires_depth"]
        ).cuda()
        label_file = os.path.join("inputs", file + ".json")
        detections = load_detections(example_dir, label_file)
        # print("HERE:", detections.bboxes) This is how to get the bounding box
        output, _ = pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"]
        )
        save_predictions(example_dir, output, file, method)

        if i == 10:
            break

    return 

def make_output_visualization(example_dir: Path, method: str) -> None:
    try:
        img_dir = os.path.join(example_dir, "images")
        for file in tqdm(sorted(os.listdir(img_dir))):
            file_path = os.path.join(img_dir, file)
            file = file.split(".")[0]

            vis_dir = example_dir / "visualizations" / method
            os.makedirs(vis_dir, exist_ok=True)
            mesh_overlay_dir = vis_dir / "mesh_overlay"
            mesh_overlay_dir.mkdir(exist_ok=True)

            contour_overlay_dir = vis_dir / "contour_overlay"
            contour_overlay_dir.mkdir(exist_ok=True)

            combined_overlay_dir = vis_dir / "combined_overlay"
            combined_overlay_dir.mkdir(exist_ok=True)

            if (
                os.path.exists(mesh_overlay_dir / "{}.png".format(file))
                and os.path.exists(contour_overlay_dir / "{}.png".format(file))
                and os.path.exists(combined_overlay_dir / "{}.png".format(file))
            ):
                print("file {} already processed".format(file))
                continue

            rgb, _, camera_data = load_observation(example_dir, file_path, load_depth=False)
            camera_data.TWC = Transform(np.eye(4))
            object_datas = load_object_data(
                example_dir / "outputs" / method / "{}.json".format(file)
            )
            object_dataset = make_object_dataset(example_dir, method)

            renderer = Panda3dSceneRenderer(object_dataset)

            camera_data, object_datas = convert_scene_observation_to_panda3d(
                camera_data, object_datas
            )
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
                render_depth=False,
                render_binary_mask=False,
                render_normals=False,
                copy_arrays=True,
            )[0]

            plotter = BokehPlotter()

            fig_rgb = plotter.plot_image(rgb)
            fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
            contour_overlay = make_contour_overlay(
                rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
            )["img"]
            fig_contour_overlay = plotter.plot_image(contour_overlay)
            fig_all = gridplot(
                [[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None
            )

            vis_dir.mkdir(exist_ok=True)
            export_png(fig_mesh_overlay, filename=mesh_overlay_dir / "{}.png".format(file))
            export_png(fig_contour_overlay, filename=contour_overlay_dir / "{}.png".format(file))
            export_png(fig_all, filename=combined_overlay_dir / "{}.png".format(file))
            logger.info(f"Wrote visualizations to {vis_dir}.")

        return
    except Exception as e:
        print(e)
        main()


def make_output_visualization_new(
    example_dir: Path,
    method: str,
    frames2process: int = None,
) -> None:
    """Make the visualizations for the output of the inference

    Args:
        example_dir (Path): Path to the example directory
        method (str): Name of the method used to recreate the mesh used for inference
        frames2process (int, optional): Maximum number of images to process. Defaults to None.
    """
    try:
        img_dir = os.path.join(example_dir, "images")
        vis_dir = example_dir / "visualizations" / method
        os.makedirs(vis_dir, exist_ok=True)
        mesh_overlay_dir = vis_dir / "mesh_overlay"
        mesh_overlay_dir.mkdir(exist_ok=True)

        contour_overlay_dir = vis_dir / "contour_overlay"
        contour_overlay_dir.mkdir(exist_ok=True)

        combined_overlay_dir = vis_dir / "combined_overlay"
        combined_overlay_dir.mkdir(exist_ok=True)

        logger.info(f"Writing visualizations to {vis_dir}.")
        for i, file in enumerate(tqdm(sorted(os.listdir(img_dir)))):
            file_path = os.path.join(img_dir, file)
            file = file.split(".")[0]

            if frames2process is not None and i >= frames2process:
                print(f"[INFO]: Stopping after {i} frames.")
                break

            if (
                os.path.exists(mesh_overlay_dir / "{}.png".format(file))
                and os.path.exists(contour_overlay_dir / "{}.png".format(file))
                and os.path.exists(combined_overlay_dir / "{}.png".format(file))
            ):
                print("[INFO]: File {} already processed".format(file))
                continue

            rgb, _, camera_data = load_observation(example_dir, file_path, load_depth=False)
            camera_data.TWC = Transform(np.eye(4))
            object_datas = load_object_data(
                example_dir / "outputs" / method / "{}.json".format(file)
            )
            object_dataset = make_object_dataset(example_dir, method)

            renderer = Panda3dSceneRenderer(object_dataset)

            camera_data, object_datas = convert_scene_observation_to_panda3d(
                camera_data, object_datas
            )
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
                render_depth=False,
                render_binary_mask=False,
                render_normals=False,
                copy_arrays=True,
            )[0]

            plotter = BokehPlotter()

            fig_rgb = plotter.plot_image(rgb)
            fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
            contour_overlay = make_contour_overlay(
                rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
            )["img"]
            fig_contour_overlay = plotter.plot_image(contour_overlay)
            fig_all = gridplot(
                [[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None
            )

            vis_dir.mkdir(exist_ok=True)
            export_png(fig_mesh_overlay, filename=mesh_overlay_dir / "{}.png".format(file))
            export_png(fig_contour_overlay, filename=contour_overlay_dir / "{}.png".format(file))
            export_png(fig_all, filename=combined_overlay_dir / "{}.png".format(file))
        return

    except Exception as e:
        print(e)
        main()

def make_output_visualization_newest(
    example_dir: Path,
    method: str,
    frames2process: int = None,
    T_W2C: Transform = Transform(np.eye(4)),
) -> None:
    """Make the visualizations for the output of the inference

    Args:
        example_dir (Path): Path to the example directory
        method (str): Name of the method used to recreate the mesh used for inference
        frames2process (int, optional): Maximum number of images to process. Defaults to None.
    """
    try:
        img_dir = os.path.join(example_dir, "images")
        vis_dir = example_dir / "visualizations" / method
        os.makedirs(vis_dir, exist_ok=True)
        mesh_overlay_dir = vis_dir / "mesh_overlay"
        mesh_overlay_dir.mkdir(exist_ok=True)

        contour_overlay_dir = vis_dir / "contour_overlay"
        contour_overlay_dir.mkdir(exist_ok=True)

        combined_overlay_dir = vis_dir / "combined_overlay"
        combined_overlay_dir.mkdir(exist_ok=True)

        logger.info(f"Writing visualizations to {vis_dir}.")

        
        object_dataset = make_object_dataset(example_dir, method)
        renderer = Panda3dSceneRenderer(object_dataset)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]
        plotter = BokehPlotter()

        for i, file in enumerate(tqdm(sorted(os.listdir(img_dir)))):
            file_path = os.path.join(img_dir, file)
            file = file.split(".")[0]

            if frames2process is not None and i >= frames2process:
                logger.info(f"Stopping after {i} frames.")
                break

            if (
                os.path.exists(mesh_overlay_dir / "{}.png".format(file))
                and os.path.exists(contour_overlay_dir / "{}.png".format(file))
                and os.path.exists(combined_overlay_dir / "{}.png".format(file))
            ):
                logger.info(f"File {file} already processed")
                continue

            rgb, _, camera_data = load_observation(example_dir, file_path, load_depth=False)

            camera_data.TWC = T_W2C
            # TW2C pro s 50 i 623 
            # m_R_w2c =np.array([0.857268, -0.510671, -0.0656292, -0.290168, -0.373897, -0.880911, 0.425317, 0.77422, -0.46871]).reshape(3,3)
            # cam_t_w2c = np.array([52.24139137959998, 43.48714846199997, 943.592320844]).reshape(3,1)
            # camera_data.TWC = Transform(cam_R_w2c, cam_t_w2c)

            object_datas = load_object_data(
                example_dir / "outputs" / method / "{}.json".format(file)
            )
            camera_data, object_datas = convert_scene_observation_to_panda3d(
                camera_data, object_datas
            )
            renderings = renderer.render_scene(
                object_datas,
                [camera_data],
                light_datas,
                render_depth=False,
                render_binary_mask=False,
                render_normals=False,
                copy_arrays=True,
            )[0]


            fig_rgb = plotter.plot_image(rgb)
            fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
            contour_overlay = make_contour_overlay(
                rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
            )["img"]
            
            fig_contour_overlay = plotter.plot_image(contour_overlay)
            fig_all = gridplot(
                [[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None
            )

            vis_dir.mkdir(exist_ok=True)
            export_png(fig_mesh_overlay, filename=mesh_overlay_dir / "{}.png".format(file))
            export_png(fig_contour_overlay, filename=contour_overlay_dir / "{}.png".format(file))
            export_png(fig_all, filename=combined_overlay_dir / "{}.png".format(file))
        return

    except Exception as e:
        print(e)
        # main()

def main():
    example_dir = Path("/home/testbed/PycharmProjects/megapose6d/local_data/examples",  global_object)

    model = "megapose-1.0-RGB-multi-hypothesis"
    # make_detections_visualization(example_dir)
    # run_inference_newest(example_dir, model, global_method, global_frames2process)
    make_output_visualization_newest(example_dir, global_method, global_frames2process)


# def main():
#     set_logging_level("info")
#     model = "megapose-1.0-RGB-multi-hypothesis"

#     for obj in objects:
#         obj_finished = {}
#         example_dir = LOCAL_DATA_DIR / "examples" / obj
#         methods = os.listdir(example_dir / "meshes")
#         print(methods)
#         for method in methods:
#             # make_detections_visualization(example_dir)
#             print(method)
#             run_inference(example_dir, model, method)
#             print("[XXXXXXXX]FINISHED INTERFERENCE")
#             make_output_visualization(example_dir, method)
#             print(":::::: VIZ")
#             obj_finished[method] = True

#         finished[obj] = obj_finished

#     print(finished)
#     with open(example_dir / "methods_done.json", "w") as f:
#         json.dump(finished, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--object", type=str, required=True)
    args.add_argument("-m", "--method", type=str, required=True)
    args.add_argument("--num_imgs", type=int, default=-1)

    args = args.parse_args()
    global global_object
    global_object = args.object
    global global_method
    global_method = args.method
    global global_frames2process
    global_frames2process = args.num_imgs if args.num_imgs > 0 else None

    main()

    # global finished
    # finished = {}
    # global objects
    # objects = ["drill"]
    # # objects = ["big_clamp", "drill", "glass", "mirror"]

    # main()
