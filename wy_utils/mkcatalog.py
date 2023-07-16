import os
import sys

# modell = "Train"#"Train"#[0-10]

# root = "../dataset/UAVid-depth_Dataset/original/China"#"../UAVid-depth_Dataset/original/China"

# txt_file = "./train.txt"

# fname = ["","image_02/data","image_03/data"]
# with open(txt_file,'w') as f:
#     files = os.listdir(os.path.join(root,modell))

#     for file in files:
#         cnt = int(file[7:])
#         if 2 == cnt or 6 == cnt:
#             fndata = fname[0]
#             img_path = os.path.join(root,modell,file,fname[0])
#         elif cnt< 18:
#             fndata = fname[1]
#             img_path = os.path.join(root,modell,file,fname[1])
#         elif cnt == 35:
#             fndata = fname[1]
#             img_path = os.path.join(root,modell,file,fname[1])
#         else:
#             fndata = fname[2]
#             img_path = os.path.join(root,modell,file,fname[2])
    

#         jpg_names = os.listdir(img_path)#os.path.join(img_path,file)
#         jpg_names.sort(key=lambda x:int(x[:-4]))
#         jpg_names = jpg_names[10:-10]
#         for jpg in jpg_names:
#             data =  modell+ "/"+ file+"/" + fndata + '/' +" " + jpg.split(".")[0] +" l"+ "\n"
#             f.write(data)

# f.close()

# modell = "Validation"#"Train"#[0-10]

# root = "../dataset/UAVid-depth_Dataset/original/China"#"../UAVid-depth_Dataset/original/China"

# txt_file = "./val.txt"

# fname = "image_03/data"
# with open(txt_file,'w') as f:
#     files = os.listdir(os.path.join(root,modell))

#     for file in files:
#         img_path = os.path.join(root,modell,file,fname)
    
#         jpg_names = os.listdir(img_path)#os.path.join(img_path,file)
#         jpg_names.sort(key=lambda x:int(x[:-4]))
#         jpg_names = jpg_names[10:-10]
#         for jpg in jpg_names:
#             data =  modell+ "/"+ file+"/" +fname+'/'+ " " + jpg.split(".")[0] +" l"+ "\n"
#             f.write(data)

# f.close()


# test

# txt_file = "./test.txt"

# with open(txt_file,'w') as f:
#     file = "Uav_1_reference_depth"
#     fname = "UAV_seq1"
#     jpg_names = os.listdir(os.path.join(root,modell,fname,))
#     jpg_names.sort(key=lambda x:int(x[:-4]))
#     jpg_names = jpg_names[1:-1]
#     for jpg in jpg_names:
#         jpg = jpg.split("_")[0]+".png"
#         data = modell+ '/'+fname + " " + jpg.split(".")[0] + " l" + "\n"
#         f.write(data)

# f.close()

####################
#----UAVself_dataset
####################
root = '/work/wy/dataset/UAVself_depth_dataset/resize_image'
with open("./splits/uav_yang_self/train.txt",'w') as f: 
    for file in os.listdir(root):
        if 'seq11'!= file:
            imgs = os.listdir(os.path.join(root,file))
            imgs.sort(key=lambda x:int(x[:-4]))
            imgs = imgs[1:-1]
            for img in imgs:
                data = file+" "+img.split('.')[0]+" "+"l"+"\n"
                f.write(data)

f.close()

root = '/work/wy/dataset/UAVself_depth_dataset/resize_image'
with open("./splits/uav_yang_self/val.txt",'w') as f: 
    for file in os.listdir(root):
        if 'seq11'== file:
            imgs = os.listdir(os.path.join(root,file))
            imgs.sort(key=lambda x:int(x[:-4]))
            imgs = imgs[1:-1]
            for img in imgs:
                data = file+" "+img.split('.')[0]+" "+"l"+"\n"
                f.write(data)

f.close()