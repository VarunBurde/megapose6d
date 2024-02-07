import csv
from megapose.config import LOCAL_DATA_DIR
import os
from tqdm import tqdm

def combine_csv(csv_files, outpath):
    header_saved = False
    total_time = 0
    time_counter = 0
    row_list = ["scene_id",	"im_id",	"obj_id",	"score",	"R",	"t",	"time"]
    with open(outpath, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer = csv.DictWriter(fout, fieldnames=row_list)
        writer.writeheader()
        for filename in csv_files:
            with open(filename, newline='') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                for e, row in enumerate(tqdm(csv_reader)):
                    if e == 0:
                        # logger.info(f"Header: {row}")
                        continue
                    scene_id, img_id, obj_id, score, Rtx, Tv, time = row
                    Rtx = Rtx.replace(","," ")
                    Tv = Tv.replace(","," ")
                    time_counter += 1
                    time = float(time)
                    total_time += time
                    writer.writerow({"scene_id": scene_id,"im_id": img_id,"obj_id": obj_id,"score":score,"R":Rtx,"t":Tv,"time": str(-1)})

    print(" average time ", total_time/time_counter)


if __name__ == '__main__':

    objects = ["01_master_chef_can", "02_cracker_box", "03_sugar_box", "04_tomatoe_soup_can", "05_mustard_bottle",
               "06_tuna_fish_can", "07_pudding_box", "08_gelatin_box", "09_potted_meat_can", "10_banana",
               "11_pitcher_base", "12_bleach_cleanser", "13_bowl", "14_mug", "15_drill", "16_wood_block", "17_scissors",
               "18_large_marker", "19_larger_clamp", "20_extra_large_clamp", "21_foam_brick"]
    csv_files = []
    outpath = LOCAL_DATA_DIR/ "examples" / "csv_results" / "combine.csv"
    for object in objects:
        print("running on object :", object)
        example_dir = LOCAL_DATA_DIR / "examples" / object
        csv_file = os.path.join(example_dir,  'ngp_results.csv')
        if os.path.exists(csv_file):
            csv_files.append(csv_file)
    combine_csv(csv_files, outpath)