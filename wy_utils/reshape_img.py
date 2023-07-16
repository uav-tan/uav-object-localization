import PIL.Image as pil
from PIL import Image
import os

root= "/work/wy/dataset/UAVself_depth_dataset/image"
files = os.listdir(root)

dir = "/work/wy/dataset/UAVself_depth_dataset/resize_image"
flag = 0
for file in files:
    for pic in os.listdir(os.path.join(root,file)):
        img = pil.open(os.path.join(root,file,pic))
        out = img.resize((640, 360),Image.ANTIALIAS)
        
        out.save(os.path.join(dir,file,"{:010d}".format(int(pic.split('.')[0]))+".jpg"))
        flag +=1
        print(flag)

# root = "/work/wy/dataset/UAVself_depth_dataset/image"
# files = os.listdir(root)
# for file in files:
#     src_path = os.path.join(root,file)
#     dst_path = os.path.join(root,'{:010d}'.format(int(file.split(".")[0][4:]))+".jpg")
#     os.rename(src_path,dst_path)