import open3d as o3d
import os
import json
path = "/home/testbed/Cluster_files/example-train/meshes_ngp"
import shutil
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

model_file = {}
old_json_file = os.path.join(path, "models_info.json")
old_data = json.load(open(old_json_file))
new_data = old_data

rotated_mesh_dir = os.path.join(path, "rotated_mesh")

for object in os.listdir(path):
    if object == "models_info.json" or object[-3:] != "obj":
        continue

    print("processing object: ", object)

    mesh = o3d.io.read_triangle_mesh(os.path.join(path, object))
    max_bound = o3d.geometry.OrientedBoundingBox.get_max_bound(mesh)
    min_bound = o3d.geometry.OrientedBoundingBox.get_min_bound(mesh)
    size_x = max_bound[0] - min_bound[0]
    size_y = max_bound[1] - min_bound[1]
    size_z = max_bound[2] - min_bound[2]

    object_id = str(int(object.split("_")[1][:-4]))

    new_data[object_id]["min_x"] = min_bound[0]
    new_data[object_id]["min_y"] = min_bound[1]
    new_data[object_id]["min_z"] = min_bound[2]
    new_data[object_id]["size_x"] = size_x
    new_data[object_id]["size_y"] = size_y
    new_data[object_id]["size_z"] = size_z

    # # rotate the mesh by 90 degrees in z axis and then 90 degrees in y axis
    # r = R.from_euler('z', 90, degrees=True)
    # mesh.rotate(r.as_matrix(), center=mesh.get_center())
    # r = R.from_euler('y', 90, degrees=True)
    # mesh.rotate(r.as_matrix(), center=mesh.get_center())
    #
    # # save the rotated mesh
    # new_mesh_path = os.path.join(rotated_mesh_dir, "object" + object_id + ".obj")
    # print("new mesh path", new_mesh_path)
    # o3d.io.write_triangle_mesh(new_mesh_path, mesh)

    if "symmetries_discrete" in new_data[object_id]:
        symentries_discite = new_data[object_id]["symmetries_discrete"]
        new_symmetry_list = []
        for i in range(len(symentries_discite)):
            symentries_dist_indi = np.array(symentries_discite[i]).reshape((4, 4))
            rotation_matrix = symentries_dist_indi[:3, :3]
            r = R.from_matrix(rotation_matrix)

            # rotate the rotation matrix by 90 degrees in z axis and then 90 degrees in y axis
            r = r * R.from_euler('z', 90, degrees=True)
            r = r * R.from_euler('y', 90, degrees=True)

            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = r.as_matrix()

            new_symmetry_list.append(transformation_matrix.flatten().tolist())
        new_data[object_id]["symmetries_discrete"] = new_symmetry_list


    if "symmetries_continuous" in new_data[object_id]:
        symentries_continuous = new_data[object_id]["symmetries_continuous"]
        new_symmetry_list = []
        for i in range(len(symentries_continuous)):
            symentries_cont_indi = symentries_continuous[i]
            axis = symentries_cont_indi["axis"]
            if axis == [0, 0, 1]:
                new_symmetry_list.append({"offset": [0, 0, 0], "axis": [0, 1, 0]})
            elif axis == [0, 1, 0]:
                new_symmetry_list.append({"offset": [0, 0, 0], "axis": [1, 0, 0]})

        new_data[object_id]["symmetries_continuous"] = new_symmetry_list


new_json_path = os.path.join(path, "models_info_new.json")
with open(new_json_path, 'w') as json_file:
    json.dump(new_data, json_file, indent=4)

    # mesh = os.path.join(path, object, "base.obj")
    #
    # obj_id = "obj_" + object.split("_")[1].zfill(6) + ".obj"
    # new_path = os.path.join(path, obj_id)
    # print(new_path)
    #
    # if not os.path.exists(new_path):
    #     # copy the mesh to the new path
    #     shutil.copy(mesh, new_path)



