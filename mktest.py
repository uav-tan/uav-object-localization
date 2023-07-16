import os
import sys

gt_path = "./seq48_ground_truth"

test_txt = "./splits/uav_yang_add/test_files.txt"
data_path = "../dataset/UAVid-depth_Dataset/original/China"
img_path = "Train/UAV_{}/image_{}/data/"
with open(test_txt, 'w') as f:
    for files in os.listdir(gt_path):
        for pic in os.listdir(os.path.join(gt_path, "depth_npy")):
            splt = pic.split("_")
            img_name = splt[1]
            img_seq = splt[0]
            f.writelines(img_path.format(img_seq, "03") + " " + img_name + " " + "l\n")
        
f.close()