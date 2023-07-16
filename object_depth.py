from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import, division, print_function

from operator import truediv

import os
import time
from ossaudiodev import SNDCTL_DSP_GETCHANNELMASK
from pickle import FALSE, TRUE
import cv2
from networks import depth_decoder
from src.opts import opt
from src.detector import Detector
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth, ScaleRecovery
from utils import readlines
import datasets
import networks

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import glob
from torchvision import transforms

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

#------------------------------
# depth predict
#------------------------------
splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.

# opt_depth

# os.environ['CUDA VISIBLE DEVICES']='1'


eval_split = 'uav_yang'

num_layers = 18

mean_flag = False #均值尺度恢复估计方法

scaling = "gt" #dgc #p4d_gt

min_depth, max_depth = 1,1000


# im_name = img_path.split("/")[-1].split(".")[0]

device = torch.device("cuda")#torch.device("cuda") 

gt_width = 640
gt_height = 352


dynamic_scale_recovery = False   #多帧图像进行尺度恢复标志位

save_npy = False # True #保存预测深度信息

obj_pos_flag = False #是否进行目标定位

total_mean_error = np.zeros(7)
total_pic_num = 0

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(paths, output_directory, load_weights_folder_path, seq_id, pic_height_info):
    """Evaluates a pretrained model using a specified test set
    """

    MIN_DEPTH = 0.001
    MAX_DEPTH = 1000

    # K = np.array([[0.58, 0, 0.5, 0],
    #               [0, 1.92, 0.5, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]], dtype=np.float32)
    
    K = np.array([[0.68, 0,-0.0069, 0],
                  [0, 1.02, -7.41758e-4, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)

    # K = np.array([[1, 0, 0, 0],
    #               [0, 1, 0, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]], dtype=np.float32)

    load_weights_folder = os.path.expanduser(load_weights_folder_path)

    assert os.path.isdir(load_weights_folder), \
        "Cannot find a folder at {}".format(load_weights_folder)

    print("-> Loading weights from {}".format(load_weights_folder))

    # filenames = readlines(os.path.join(splits_dir, eval_split, "my_pic_files.txt"))
    encoder_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    img_ext = '.jpg'

    # dataset = datasets.UAVDataset(data_path, paths,
    #                                     encoder_dict['height'], encoder_dict['width'],
    #                                     [0], 4, is_train=False, img_ext=img_ext)
    # dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=12,
    #                         pin_memory=True, drop_last=False)

    encoder = networks.ResnetEncoder(num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()

    pred_disps = []
    pred_img_dict = dict()
    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))



    with torch.no_grad():
        # for data in dataloader:
        #     input_color = data[("color", 0, 0)].cuda()
        #     output = depth_decoder(encoder(input_color))

        #     pred_disp, _ = disp_to_depth(output[("disp", 0)], min_depth, max_depth)
        #     pred_disp = pred_disp.cpu()[:, 0].numpy()

        #     pred_disps.append(pred_disp)
        for image_path in paths: #处理多张图像 
            # Load image and preprocess
            im_name = image_path.split("/")[-1]
            input_image = pil.open(image_path).convert('RGB')
            # original_width, original_height = input_image.size
            input_image = input_image.resize((640, 352)) #, pil.Resampling.LANCZOS
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
            pred_disp, _ = disp_to_depth(outputs[("disp", 0)], min_depth, max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            pred_disps.append(pred_disp)
            pred_img_dict[im_name] = pred_disp

    pred_disps = np.concatenate(pred_disps)

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling: {}".format(scaling))
    ratios = []
    errors = []

    #加载序列对应的相机参数
    cam_calib = []
    with open("./{}/{}_ground_truth/{}_cam_calibrated.txt".format(Test_path, seq_id, seq_id),'r') as f:
        dta = f.readlines()
        for d in dta:
            cam_calib.append(d.split())
    
    tensor_K = np.array([[float(cam_calib[0][0]), 0,float(cam_calib[0][2])],
                  [0, float(cam_calib[1][1]), float(cam_calib[1][2])],
                  [0, 0, 1]], dtype=np.float32)
    tensor_K = torch.from_numpy(tensor_K).unsqueeze(0).cuda()
    # for i in range(pred_disps.shape[0]):
    res_pre_depth_dict = dict()

    if (dynamic_scale_recovery): #多帧图像进行尺度恢复 
        # 选取多帧图像的地面点共同进行ransac优化算法 得到更加准确的地面点 
        # 输入的图像必须大于等于三张
        if (os.path.isfile(img_path)):
            raise Exception("输入图像数目必须大于三张")
        
        keys = list(pred_img_dict)
        for i in range(1, len(keys) - 1):
            pred_disp = pred_img_dict[keys[i]]
            # pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp[0],(640,360))
            pred_depth = 1 / pred_disp

            gt_depth= np.load(gt_depth_paths.format(seq_id, seq_id,keys[i].split('.')[0]))
            gt_height, gt_width = gt_depth.shape[:2]
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            if scaling == "dgc":

                #加载对应的相机真实高度
                cam_height_in = pic_height_info[keys[i]][0]  #第i帧图像的真值高度
                cam_height = torch.tensor([cam_height_in]).cuda()
                scale_recovery = ScaleRecovery(1, gt_height, gt_width).cuda() #创建尺度恢复对象
                pred_depth_multi = []
                img_path_seg = []
                for j in range(-1,1): # i -1 帧 i帧 i+1帧图像处理
                    pred_disp = pred_img_dict[keys[i + j]]
                    # pred_disp = pred_disps[i]
                    pred_disp = cv2.resize(pred_disp[0],(640,360))

                    pred_depth_bef = 1 / pred_disp
                    pred_depth_bef = torch.from_numpy(pred_depth_bef).unsqueeze(0).cuda()
                    pred_depth_multi.append(pred_depth_bef) #把三个处理好的深度放进去
                    img_path_seg.append(os.path.join(img_path,keys[i + j]))
                
                ratio = scale_recovery(pred_depth_multi, tensor_K, cam_height,img_path_seg, dynamic_scale_recovery).cpu().item() # 
                pred_depth = pred_depth_multi[1][0].cpu().numpy()
                
            if scaling == "gt":
                #eigen split
                # crop = np.array(
                #     [0.40810811 * gt_height, 0.99189189 * gt_height,
                #      0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                # crop_mask = np.zeros(mask.shape)
                # crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                # mask = np.logical_and(mask, crop_mask)
                ratio = np.median(gt_depth) / np.median(pred_depth)
            
            # print("ratio: {}".format(ratio))
            pred_depth *= ratio #1?
            ratios.append(ratio)

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            if len(gt_depth) != 0:
                errors.append(compute_errors(gt_depth[mask], pred_depth[mask]))

            # # save pre_depth value
            if save_npy:
                np.save(os.path.join(output_directory,'{}.npy'.format(keys[i])), pred_depth) #key
            
            # Saving colormapped depth image
            pred_depth_np = 1/pred_depth
            vmax = np.percentile(pred_depth_np, 95)
            normalizer = mpl.colors.Normalize(vmin=pred_depth_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(pred_depth_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory,'{}_depth.jpg'.format(keys[i]))
            im.save(name_dest_im)
            res_pre_depth_dict[keys[i]] = pred_depth
    else:  # 默认 单张图像尺度恢复
        for key in pred_img_dict:
            pred_disp = pred_img_dict[key]
            # pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp[0],(640,360))
            pred_depth = 1 / pred_disp
            
            gt_depth= np.load(gt_depth_paths.format(seq_id, seq_id,key.split('.')[0]))
            gt_height, gt_width = gt_depth.shape[:2]
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            if scaling == "dgc":
                
                # tensor_K = K.copy()
                # tensor_K[0, :] *= gt_width
                # tensor_K[1, :] *= gt_height
                

                #加载对应的相机真实高度
                cam_height_in = pic_height_info[key][0]

            

                cam_height = torch.tensor([cam_height_in]).cuda()
                # cam_height = torch.tensor([52]).cuda() #均值版本 自定义高度
                scale_recovery = ScaleRecovery(1, gt_height, gt_width).cuda()
                pred_depth = torch.from_numpy(pred_depth).unsqueeze(0).cuda()   
                
                if os.path.isdir(img_path):
                    img_path_seg = os.path.join(img_path,key)
                if os.path.isfile(img_path):
                    img_path_seg = img_path
                if mean_flag:
                    ratio = gt_depth[mask].mean() / pred_depth[0][mask].cpu().numpy().mean() # 均值
                    # ratio = np.median(gt_depth) / np.median(pred_depth.cpu().numpy()) #中值
                else:
                    ratio = scale_recovery(pred_depth, tensor_K, cam_height,img_path_seg).cpu().item() # 
                pred_depth = pred_depth[0].cpu().numpy()

            if scaling == "gt":
                #eigen split
                # crop = np.array(
                #     [0.40810811 * gt_height, 0.99189189 * gt_height,
                #      0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                # crop_mask = np.zeros(mask.shape)
                # crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1mean
                # mask = np.logical_and(mask, crop_mask)
                ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
            
            # print("ratio: {}".format(ratio))

            pred_depth *= ratio #1?
            ratios.append(ratio)

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            if len(gt_depth) != 0:
                errors.append(compute_errors(gt_depth[mask], pred_depth[mask]))

            # # save pre_depth value
            if save_npy:
                np.save(os.path.join(output_directory,'{}.npy'.format(key)), pred_depth)
            
            # Saving colormapped depth image
            pred_depth_np = 1/pred_depth
            vmax = np.percentile(pred_depth_np, 95)
            normalizer = mpl.colors.Normalize(vmin=pred_depth_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(pred_depth_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory,'{}_depth.jpg'.format(key))
            im.save(name_dest_im)
            res_pre_depth_dict[key] = pred_depth

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))


    mean_errors = np.array(errors).mean(0)
    
    global total_mean_error, total_pic_num
    total_mean_error += mean_errors
    total_pic_num += len(errors)
    print("\n图像序列:{}, 图像总数:{}".format(seq_id, len(errors)))

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    
    return res_pre_depth_dict

#------------------------------
# object detection
#------------------------------

dataset_info = {
            "num_classes": 10,
            "mean": np.array([0.37294899, 0.37837514, 0.36463863],
                                dtype=np.float32).reshape(1, 1, 3),
            "std": np.array([0.19171683, 0.18299586, 0.19437608],
                            dtype=np.float32).reshape(1, 1, 3),
            "class_name": ['pedestrian', 'people', 'bicycle', 'car',
                            'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'],
            "valid_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
vis_thresh = 0.4


def get_depth(depth,bbox,shape):
    # center_x = int((bbox[0]+bbox[2])/2)
    # center_y = int((bbox[1]+bbox[3])/2)
 
    if "p4d_gt" == scaling:
        bbox[1] = int(bbox[1])
        bbox[3] = int(bbox[3])
        bbox[2] = int(bbox[2])
        bbox[0] = int(bbox[0])
        dp1 = depth[bbox[1]][bbox[0]]
        dp2 = depth[bbox[3]][bbox[2]]
        
        if dp1 !=0 and dp2 != 0:
            obj_depth = (dp1+dp2)/2
        else:
            ii = np.linspace(bbox[1],bbox[3],8)
            jj = np.linspace(bbox[0],bbox[2],8)
            cnt = 0
            sum = 0
            for i in ii:
                for j in jj:
                    if depth[int(i),int(j)] != 0:
                        cnt+=1
                        sum+=depth[int(i),int(j)]
            if 0 == cnt:
                obj_depth = 0
            else:
                obj_depth = sum/cnt


    else:
        bbox[1] = int(bbox[1]/shape[0]*352)
        bbox[3] = int(bbox[3]/shape[0]*352)
        bbox[2] = int(bbox[2]/shape[1]*640)
        bbox[0] = int(bbox[0]/shape[1]*640)
        dp1 = depth[bbox[1]][bbox[0]]
        dp2 = depth[bbox[3]][bbox[2]]
        obj_depth = (dp1+dp2)/2#depth[center_y][center_x]

    return obj_depth


def get_obj_pos(path, depth,out_path): #result 为目标检测的结果
    detector = Detector(opt)
    
    image = cv2.imread(path)
    im_name = path.split("/")[-1].split(".")[0]
    ret = detector.run(image)   
    if scaling == 'gt':
        result_path = os.path.join(out_path,'{}_obj_gt.jpg'.format(im_name))
    if scaling == 'dgc':
        result_path = os.path.join(out_path,'{}_obj_dgc.jpg'.format(im_name))
    if scaling == 'p4d_gt':
        result_path = os.path.join(out_path,'{}_obj_p4d_gt.jpg'.format(im_name))
    results = ret['results']
    
    for cls_ind in range(1, dataset_info["num_classes"] + 1):
        for bbox in results[cls_ind]:
            conf = bbox[4]
            # filter low score
            if conf < vis_thresh:
                continue
            bbox = np.array(bbox[:4], dtype=np.int32)

            class_name = dataset_info["class_name"]

            cv2.rectangle(img=image,
                        pt1=(bbox[0], bbox[1]),
                        pt2=(bbox[2], bbox[3]),
                        color=[0, 0, 255],
                        thickness=1)
            #txt
            cv2.putText(img=image,
                        text=f'{class_name[cls_ind-1]}|d:{get_depth(depth,bbox,image.shape[0:2]):.1f}',
                        org=(bbox[0], bbox[1] - 2),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.3,
                        color=(0, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)

    cv2.imwrite(result_path, image)

def main(paths, output_directory, load_weights_folder_path, seq_id, obj_pos_flag, pic_height_info):
    
    if "p4d_gt" == scaling:
        depth = np.load(gt_depth_paths)#还需要format 视频序列和图像名字
    else:
        res_pre_depth_dict = evaluate(paths, output_directory, load_weights_folder_path, seq_id, pic_height_info) #获取深度估计结果
    
    if (obj_pos_flag): #进行目标定位
        for key in res_pre_depth_dict: 
            print("get_obj_pos:{}|seqID:{}".format(key,seq_id))
            if os.path.isdir(img_path):
                path = os.path.join(img_path,key)
            if os.path.isfile(img_path):
                path = img_path
            pred_depth = res_pre_depth_dict[key]
            get_obj_pos(path, pred_depth, output_directory) #传入单张图片的路径和预测的深度信息

    print("FINISH!")


if __name__ == '__main__':
    weight_dirs = open('weight_path.txt').readlines() # weight_path copy.txt   weight_path.txt
    for weight_name in weight_dirs:
        # 单张图像
        total_mean_error = np.zeros(7)
        total_pic_num = 0
        # img_path = "./{}_ground_truth/depth_img".format(seq_id)#'./seq48_ground_truth/seq48_build/0000000321.jpg'#'./0000000001.jpg'#'0000027811.jpg' #'./seq49_build/0000000161.jpg'
        # gt_depth_paths = 
        print("====================================================================================================================================================")
        print("====================begin==========================")
        print("time: {}".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))) 
        print("===================================================")
        if dynamic_scale_recovery:
            print("！！！使用多帧尺度恢复！！！")
        Test_path = "Test_dataset" #
    
        load_weights_folder_path = os.path.join('~/work/wy/DNet-master/', weight_name.strip("\n"))
        for seq_name in os.listdir(Test_path):
            seq_id = seq_name.split('_')[0]
            # 加载的权重文件夹weight_path copy.txt
            img_path = "./{}/{}_ground_truth/build".format(Test_path, seq_id)#"./seq48_ground_truth/build/0000000321.jpg"##'./seq48_ground_truth/seq48_build/0000000321.jpg'#'./0000000001.jpg'#'0000027811.jpg' #'./seq49_build/0000000161.jpg'
            # FINDING INPUT IMAGES

            if os.path.isfile(img_path): #单张图像
                # Only testing on a single image
                paths = [img_path]
                gt_depth_paths = "./{}/{}_ground_truth/depth_npy/{}_{}_gt_depth.npy".format(Test_path, seq_id, seq_id, img_path.split('/')[-1].split('.')[0])
                # output_directory =  "./output"#os.path.dirname(args.image_path)
                output_directory = os.path.join("lib_output", weight_name.split('/')[1], weight_name.split('/')[-1].strip("\n"), seq_name) #"./results-src"
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
            elif os.path.isdir(img_path): #多张图像 文件夹
                # Searching folder for images
                paths = glob.glob(os.path.join(img_path, '*.jpg'))
                paths.sort(key=lambda x:int(x[-7:-4]))
                gt_depth_paths = "./" + Test_path + "/{}_ground_truth/depth_npy/{}_{}_gt_depth.npy"

                output_directory = os.path.join("lib_output", weight_name.split('/')[1], weight_name.split('/')[-1].strip("\n"), seq_name) #"./results-src"
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
            else:
                raise Exception("Can not find args.image_path: {}".format(img_path))
            

            print("-> Predicting on {:d} test images, seq ID:{}".format(len(paths), seq_id))

            # 根据视频序列和图片名称读取相机高度 
            pic_height_info = dict()

            with open("./{}/{}_ground_truth/{}_cam_height.txt".format(Test_path, seq_id, seq_id),'r') as f:
                cam_height_info = f.readlines()
                for img_name_height in cam_height_info: # 读取相机对应高度
                    # if (img_name_height.split()[0] == img_path.split('/')[-1]):
                    cam_height_in = (abs(float(img_name_height.split()[1])))#74.1169 #54
                    pic_height_info[img_name_height.split()[0]] = [cam_height_in]

            
            main(paths, output_directory, load_weights_folder_path, seq_id, obj_pos_flag, pic_height_info)


        print("\n 图像总数:{}".format(total_pic_num))
        total_mean_error /= 2
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*total_mean_error.tolist()) + "\\\\")
        print("---------------------------------------------------------------------------------------------------")
        # time_str = ''
        # for stat in ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']:
        #     time_str += f'{stat} {ret[stat]:.3f}s |'
        # print(time_str)



