import csv
from tqdm import tqdm

if __name__ == '__main__':
    path2csv_file = "/home/testbed/PycharmProjects/megapose6d/local_data/examples/02_cracker_box/refiner-final_ycbv-test.csv"
    with open(path2csv_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for e, row in enumerate(tqdm(csv_reader)):
            if e == 0:
                # logger.info(f"Header: {row}")
                continue
            # scene_id, img_id, obj_id, score, Rtx, Tv, time = row
