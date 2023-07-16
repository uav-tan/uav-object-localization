from syslog import setlogmask
import numpy as np
import PIL.Image as pil

from pyexiv2 import Image
import math
import os
import cv2
# root = './wy_utils/0000_gt.jpg'
# out_dir = './'

# img = Image(root)
# xmp_info = img.read_xmp()
# exif_info = img.read_exif()

# uav_rAlt = float(xmp_info['Xmp.drone-dji.RelativeAltitude'])#60.036

# depth = np.load("./out_341.npy")# (176,320)


# 39.13313236
# 117.03848949
# def out():
#     print("111")

# factor =1 #260
# # factor = 60.036/6.1860504
# # 6.1860504
# gt_depth = depth*factor
# print("1")

# out()
# gt_dis = gt_depth[273][328]  # 328 273
# print(gt_dis)

# img = pil.open("./output/0000000348_depth.png")
# img = img.convert('L') 
# img = np.array(img)/255
# print(img)


# segment = np.load('out.npy')
# mask = np.where(10 == segment,1,0)
# print(mask)


# npy读取转xyz文件 

# cam_point = np.load('out_cam_point_461.npy')
# xyz_path = "./out_cam_point_461.xyz"
# with open(xyz_path, 'w') as f:
#     for dta in cam_point:
#         f.write(str(dta[0]) + " " + str(dta[1]) + " " + str(dta[2]) + "\n")

# f.close()
# npy读取转xyz文件  END

# npy cam_point点筛除
cam_point = np.load('out_cam_point_461.npy')
cam_filter_res = []
for dta in cam_point:
    if dta[0] == 0 and dta[1] == 0 and dta[2] == 0:
        continue
    cam_filter_res.append(dta)

np.save('./out_cam_point_461.npy', np.array(cam_filter_res))

# npy cam_point点筛除 END