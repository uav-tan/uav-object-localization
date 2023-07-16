from __future__ import absolute_import, division, print_function
import math

from ransac_DNet import RANSAC_filter
from gp_ransac import ransac_ground_plane_optimization

# from segmentation import get_segment_mask

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as pil
import os
import random

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords),
            requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
                       requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1).reshape(
            self.batch_size, 4, self.height, self.width) 

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        points = points.view(self.batch_size, 4, -1)
        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x, sf=2):
    """Upsample input tensor by a factor
    """
    return F.interpolate(x, scale_factor=sf, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)



class GroundPointOptimize(): #地面点优化部分
    def __init__(self, data, goal_inliers, max_iterations,):
        super(GroundPointOptimize, self).__init__()
        self.ground_point = data
        self.goal_inliers = goal_inliers
        self.max_iterations = max_iterations
        self.sample_size = 3

    def augment(self, xyzs):
        axyz = np.ones((len(xyzs), 4))
        axyz[:, :3] = xyzs
        return axyz

    def estimate(self, xyzs):
        axyz = self.augment(xyzs[:3])
        return np.linalg.svd(axyz)[-1][-1, :]

    def is_inlier(self, coeffs, xyz, threshold):
        return np.abs(coeffs.dot(self.augment([xyz]).T)) < threshold

    def run_ransac(self, stop_at_goal=True, random_seed=None):
        best_ic = 0
        best_model = None
        random.seed(random_seed)
        # random.sample cannot deal with "data" being a numpy array
        data = list(self.ground_point)
        for i in range(self.max_iterations):
            s = random.sample(data, int(self.sample_size))
            m = self.estimate(s)
            ic = 0
            for j in range(len(data)):
                if self.is_inlier(m, data[j], 0.01):
                    ic += 1

            print(s)
            print('estimate:', m,)
            print('# inliers:', ic)

            if ic > best_ic:
                best_ic = ic
                best_model = m
                if ic > self.goal_inliers and stop_at_goal:
                    break
        print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
        return best_model, best_ic


class ScaleRecovery(nn.Module):
    """Layer to estimate scale through dense geometrical constrain
    """
    def __init__(self, batch_size, height, width):
        super(ScaleRecovery, self).__init__()
        self.backproject_depth = BackprojectDepth(batch_size, height, width)
        self.batch_size = batch_size
        self.height = height
        self.width = width

    # derived from https://github.com/zhenheny/LEGO
    def get_surface_normal(self, cam_points, nei=1):
        cam_points_ctr  = cam_points[:, :-1, nei:-nei, nei:-nei]
        cam_points_x0   = cam_points[:, :-1, nei:-nei, 0:-(2*nei)]
        cam_points_y0   = cam_points[:, :-1, 0:-(2*nei), nei:-nei]
        cam_points_x1   = cam_points[:, :-1, nei:-nei, 2*nei:]
        cam_points_y1   = cam_points[:, :-1, 2*nei:, nei:-nei]
        cam_points_x0y0 = cam_points[:, :-1, 0:-(2*nei), 0:-(2*nei)]
        cam_points_x0y1 = cam_points[:, :-1, 2*nei:, 0:-(2*nei)]
        cam_points_x1y0 = cam_points[:, :-1, 0:-(2*nei), 2*nei:]
        cam_points_x1y1 = cam_points[:, :-1, 2*nei:, 2*nei:]

        vector_x0   = cam_points_x0   - cam_points_ctr
        vector_y0   = cam_points_y0   - cam_points_ctr
        vector_x1   = cam_points_x1   - cam_points_ctr
        vector_y1   = cam_points_y1   - cam_points_ctr
        vector_x0y0 = cam_points_x0y0 - cam_points_ctr
        vector_x0y1 = cam_points_x0y1 - cam_points_ctr
        vector_x1y0 = cam_points_x1y0 - cam_points_ctr
        vector_x1y1 = cam_points_x1y1 - cam_points_ctr

        normal_0 = F.normalize(torch.cross(vector_x0,   vector_y0,   dim=1), dim=1).unsqueeze(0)
        normal_1 = F.normalize(torch.cross(vector_x1,   vector_y1,   dim=1), dim=1).unsqueeze(0)
        normal_2 = F.normalize(torch.cross(vector_x0y0, vector_x0y1, dim=1), dim=1).unsqueeze(0)
        normal_3 = F.normalize(torch.cross(vector_x1y0, vector_x1y1, dim=1), dim=1).unsqueeze(0)

        normals = torch.cat((normal_0, normal_1, normal_2, normal_3), dim=0).mean(0)
        normals = F.normalize(normals, dim=1)

        refl = nn.ReflectionPad2d(nei)
        normals = refl(normals)

        return normals

    def get_ground_mask(self, cam_points, normal_map, img_path,threshold=5):#5
        b, _, h, w = normal_map.size()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        threshold = math.cos(math.radians(threshold))
        ones, zeros = torch.ones(b, 1, h, w).cuda(), torch.zeros(b, 1, h, w).cuda()
        theta = 90
       
        angles_cos = ones*math.cos(theta)
        angles_sin = ones*math.sin(theta)
        vertical = torch.cat((zeros, angles_cos, angles_sin), dim=1)

        # # # vertical = torch.cat((zeros, zeros,ones), dim=1) # DNET 方法
        cosine_sim = cos(normal_map, vertical).unsqueeze(1)
        vertical_mask = (cosine_sim > threshold) | (cosine_sim < -threshold)

        y = cam_points[:,1,:,:].unsqueeze(1)
        ground_mask = vertical_mask.masked_fill(y <= 0, False)

        # # end DNet方法-------------------------------------------------------
        # lab_mask = np.zeros(ground_mask.size()[2:])
        # lab_mask[38:352,290:341]=1

        # -------------------------------------------- 原来的分割代码
        # segment = get_segment_mask(img_path) # 原来的分割代码
      
       
        # segment = np.load('out_341.npy')

        # lab_mask = np.where(10 == segment,1,0)
        # ground_mask = torch.from_numpy(lab_mask.astype(bool)) #分割的mask
        # -------------------------------------------- 原来的分割代码 END
        # # #-----------------------GeoSeg分割结果-------------------------
        # splited = img_path.split('/')
        # lab_mask = np.load(os.path.join(splited[1], splited[2],"mask/npy/Labels",splited[-1].split('.')[0] + '.npy'))
        # ground_mask = torch.from_numpy(lab_mask.astype(bool)) #分割的mask
        # # # # #-----------------------GeoSeg分割结果------------------------- END

        # dta = (lab_mask*255).astype(np.uint8)

        # # dta = (ground_mask.cpu().numpy()[0][0]*255).astype(np.uint8)
        # im = pil.fromarray(dta)
        # # im.save("./mask.jpg")d

        return ground_mask

    def forward(self, depth, K, real_cam_height,img_path, dynamic_scale_recovery=False):
        inv_K = torch.inverse(K)
        # RANSAC 尺度恢复方法
        if (dynamic_scale_recovery): #使用滑动窗口法同时处理三张图像的cam_point
            cam_point_cut = []
            for i in range(0, len(img_path)):
                cam_point = self.backproject_depth(depth[i], inv_K)
                surface_normal = self.get_surface_normal(cam_point)
                ground_mask = self.get_ground_mask(cam_point, surface_normal,img_path[i]) # campoint传入三个 img_path传入三个
                segemented_cam_point = cam_point * ground_mask.cuda() #掩膜筛选后的地面点集合 
                cam_point_cut.append(segemented_cam_point[0,:-1,:,:].cpu().numpy())
        else:#正常的处理逻辑
            cam_points = self.backproject_depth(depth, inv_K)
            surface_normal = self.get_surface_normal(cam_points)
            ground_mask = self.get_ground_mask(cam_points, surface_normal,img_path) # campoint传入三个 img_path传入三个
            segemented_cam_point = cam_points * ground_mask.cuda() #掩膜筛选后的地面点集合 
            cam_point_cut = segemented_cam_point[0,:-1,:,:].cpu().numpy()

        # RANSAC 尺度恢复方法 end -------------------------------------5
        #TODO step 地面点优化 自己写的：：：
        
        # segemented_cam_point = segemented_cam_point[0,0:3,:,:]
        # out_segmented = segemented_cam_point[0,:-1,:,:].reshape(-1,3).cpu().numpy() #输出地面淹没分割后的三维点 reshape结果是错的
        
        
        # # ## RANSAC open3d -------
        if (type(cam_point_cut) == np.ndarray):
            C, H, W = cam_point_cut.shape
            out_cam_point = []
            for i in range(0, H):
                for j in range(0, W):
                    out_cam_point.append(cam_point_cut[:,i,j]) 
        if (type(cam_point_cut) == list): #多帧 多地面点构造部分5
            out_cam_point = []
            for ot_cam_pt in cam_point_cut:
                C, H, W = ot_cam_pt.shape
                for i in range(0, H):
                    for j in range(0, W):
                        out_cam_point.append(ot_cam_pt[:,i,j]) 

        out_cam_point = np.array(out_cam_point)
        ##------------------------------

        # np.save('./out_cam_point_{}.npy'.format(img_path.split('/')[-1].split(".")[0]), out_cam_point) #demo测试 保存文件
        # #     ## RANSAC open3d END END

        # RANSAC version 1
        # out_segmented = cam_point_cut.cpu() # t().

        # # 原始版本
        # out_segmented_fliter = out_cam_point[np.flatnonzero(out_cam_point[:,0]),:] #筛选其中0元素
        # num_point = out_segmented_fliter.shape[0] #筛选过后的地面点 
        
        # RANSAC地面点优化算法版本1 效果一般
        # GPO = GroundPointOptimize(out_segmented_fliter, num_point * 0.5, 30) # max_iterations 20
        

        # best_model, _ = GPO.run_ransac() #best_model为平面方程的一般形式 Ax + By + Cz + D = 0 中的 A B C D 平面法向量为 (A, B, C) 【非单位法向量】

        # # # TODO
        # A, B, C, D = best_model
        # sub = math.sqrt(A**2 + B**2 + C**2)
        # sub_norm_surface = torch.from_numpy(np.array([A/sub, B/sub, C/sub])).cuda()
        # height = 2.4993004353962034
        ##end
       

        # n = 100
        # max_iterations = 100
        # goal_inliers = n * 0.8

        # xyzs = np.random.random((n, 3)) * 10
        # xyzs[:50, 2:] = xyzs[:50, :1]

        # GPO = GroundPointOptimize(xyzs, n * 0.3, 20) 

        # best_model, _ = GPO.run_ransac()

        # end

        # cam_heights = (cam_points[:,:-1,:,:] * surface_normal).sum(1).abs().unsqueeze(1)
        # cam_heights = (cam_points[:,:-1,:,:] * sub_norm_surface).sum(1).abs().unsqueeze(1)
        # RANSAC# RANSAC version 11 效果一般 END END
        
        # 原始版本：：：：DNet 
        # cam_heights = (cam_points[:,:-1,:,:] * surface_normal).sum(1).abs().unsqueeze(1).cuda() # 除去一维度的标准化1

        # 使用筛选后的地面mask版本：
        # # # DNet 版本！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # cam_heights = (segemented_cam_point[:,:-1,:,:] * surface_normal).sum(1).abs().unsqueeze(1) # 除去一维度的标准化1

        # cam_heights_masked = torch.masked_select(cam_heights, ground_mask.cuda())
        # cam_height = torch.mean(cam_heights_masked).unsqueeze(0)
        # # # DNet 版本！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ END

        # # # mean 版本！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # cam_heights = (segemented_cam_point[:,:-1,:,:] * surface_normal).sum(1).abs().unsqueeze(1) # 除去一维度的标准化1

        # cam_heights_masked = torch.masked_select(cam_heights, ground_mask.cuda())
        # cam_height = torch.mean(cam_heights_masked).unsqueeze(0)
        # # # mean 版本！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ END

        #  # 均值mask版本 ----------------------高度52----------- END
        # # 使用筛选后的地面mask版本：END
        # # cam_height = torch.tensor([0.37087111]).cuda()
        # height = 1.4876563965066578
        # cam_height = torch.tensor([height]).cuda()


        # -----------------------------------------RANSAC 地面点优化程序 yang1.0-----------------------------------------------
         # 分割地面点筛选过的cam_point  #"./out_cam_point_461.npy"
        RanFilter = RANSAC_filter()

        pcd = RanFilter.read_txt(out_cam_point)#将筛选0点后的地面点集送入RANSAC#, row_skip=1, split_char=' ')out_cam_point
        pcd = pcd[:,:3]
        # # 第三个参数：maxdst = 0.5
        point_size = len(pcd)
        plane_set, plane_inliers_set = RanFilter.ransac_plane_detection(pcd, 3, 0.1, stop_inliers_ratio=0.95, max_trials= 100,  initial_inliers=None,
                                        out_layer_inliers_threshold=point_size*0.3, out_layer_remains_threshold=point_size*0.3) #max_trials=1000 #, data_remains
        plane_set = np.array(plane_set)


        # # #------------------gpt ransac plane --------------------------
        # res = ransac_ground_plane_optimization(pcd, 0.1, 100)
        # A, B, C, D = res
        # dist = abs(D)/math.sqrt(A**2 + B**2 + C**2)
        # # -------------------------------------------
        # print("================= 平面参数 ====================")
        # print(plane_set) #（cx,cy,cz,nx,ny,nz)
        # 绘图 平面模型参数（平面上任意一点cX,cy,cX)+(平面法向量nx,ny,nz)
        # show_3dpoints(plane_inliers_set)
        # print("cam_point:{}".format(img_path))
        # print("================= 远点到平面的距离 ====================")
        # x0, y0, z0, A, B, C = plane_set[0]
        # D = (A*(x0) + B*(y0) + C*(z0))
        # # # 新的平面方程 
        A, B, C = plane_set
        D = np.dot(plane_set, plane_inliers_set)
        # # 新的平面方程 END
        dist = abs(D)/math.sqrt(A**2 + B**2 + C**2)
        # # dist = x0*A + y0*B + z0*C 
        # # print("相机坐标系下远点到平面的距离：{}".format(dist))
        # # print("图像路径：\n{}".format(img_path))

        cam_height = torch.tensor([dist]).cuda()
        # -----------------------------------------RANSAC 地面点优化程序 yang1.0-----------------------------------------------END
        

        scale = torch.reciprocal(cam_height).mul_(real_cam_height)
        # print("分割mask后相机坐标系下相机高度{}, 恢复尺度因子{},真实相机高度{}".format(cam_height.item(), scale.item(),real_cam_height.item()))
        return scale


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
