#! /usr/bin/bash

# This script runs inference and visualization for selected objects.
# The objects are located in data/examples. 
# Script requres the following structure inside data/examples/<object>:
#   camera_data.json - camera parameters
#   /images - folder with images
#   /inputs - folder with json files with corresponding bounding boxes for each image
#   /meshes - folder with folder for each method with corresponding meshes
# The output videos are saved in data/examples/<object>/videos/<object_method>.mp4
# The output images are saved in data/examples/<object>/visualizations/<method>/combined_overlay

# Description of the variables:
#   objects - array of objects to process in format (object1 object2 object3 ...) for single object (object1)
#   fps - frames per second for the output video
#   num_imgs - number of images to process. If -1, all images are processed
#   make_video - if true, the script will create videos from the output images
#   force_video - if 0, then the script will not overwrite existing videos else it will overwrite them

start_time=$(date +%s)


object_dirs=data/examples
#objects=(drill big_clamp glass mirror) #big_clamp) # Into objects array write the names of the objects as they are in data/examples
# objects=(Axle_rear Chassis_1 Chassis_2 Chassis_3) #3D_base-bigbbox  3D_wheels
# objects=(Axle_rear-Varun)
# objects=(Chassis_1-bbox_test)
# objects=(Motor Servo Battery Controller Main)
objects=(Drill)
finished=()
fps=30
num_imgs=-1
make_video=false
force_video=0

for object in ${objects[@]}; do
    echo $object
    method_dirs=$object_dirs/$object/meshes/*

    if $make_video; then 
        video_dir=$object_dirs/$object/videos/
        mkdir -p $video_dir
    fi 

    mkdir -p $object_dirs/$object/visualizations
    mkdir -p $object_dirs/$object/outputs
    for method_dir in $method_dirs; do
        method=$(basename $method_dir)
        echo $method
        echo $method_dir
        img_folder=$object_dirs/$object/visualizations/$method/combined_overlay
        input_folder=$object_dirs/$object/inputs
        video_name=$object\_$method.mp4
        /home/varun/anaconda3/envs/megapose/bin/python /home/varun/PycharmProjects/megapose6d_varun/src/megapose/scripts/run_inference_multiple_methods.py -m $method -o $object --num_imgs $num_imgs
        if $make_video; then
            /home/varun/anaconda3/envs/megapose/bin/python /home/varun/Projects/camera_calibration/images2video.py --method $method  --fps $fps  --img_folder $img_folder --input_folder $input_folder --output_folder $video_dir --video_name $video_name  --force_overwrite $force_video
        fi
        finished+=($method_dir)
    done
done
echo Finished: 
for fin in ${finished[@]}; do
    echo \  $fin
done
end_time=$(date +%s)
echo Execution time was $(($end_time - $start_time)) seconds.
