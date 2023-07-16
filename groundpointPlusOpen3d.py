# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类
import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from mpl_toolkits.mplot3d import Axes3D
import random
import open3d
import math
import sys
from scipy.spatial import KDTree
import sklearn.cluster

#读取bin文件获取三维数据
def read_velodyne_bin(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

#最小二乘函数，优化提取的平面
#输入参数:待优化的点云数
#输出参数:最小二乘拟合的平面参数(a,b,c)
def PlaneLeastSquare(X: np.ndarray):
    # z=ax+by+c,return a,b,c
    A = X.copy()
    b = np.expand_dims(X[:, 2], axis=1)  #(M,)tuple类型变成 M*1的array类型
    A[:, 2] = 1
    # 通过X=(AT*A)-1*AT*b直接求解
    A_T = A.T
    A1 = np.dot(A_T, A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, A_T)
    x = np.dot(A3, b)
    return x

#RANSAC算法，提取平面
#输入参数 X:输入的点云数据，tao:外点距离平面的距离因子,e:一个点是外点的概率,N_regular:默认迭代次数
#输出参数:内点，外点，平面的四参数(a,b,c,d)
def PlaneRANSAC(X: np.ndarray, tao: float, e=0.4, N_regular=100):
    # return plane ids
    t = X.shape[0]
    s = X.shape[1]
    count = 0
    p = 0.99
    dic = {}
    #确认迭代次数
    if math.log(1 - (1 - e) ** s) < sys.float_info.min:
        N = N_regular
    else:
        N = math.log(1 - p) / math.log(1 - (1 - e) ** s)

    # 开始迭代
    while count < N:
        ids = random.sample(range(0, t), 3)
        p1, p2, p3 = X[ids]
        # 判断是否共线
        L = p1 - p2
        R = p2 - p3
        if 0 in L or 0 in R:
            continue
        else:
            if L[0] / R[0] == L[1] / R[1] == L[2] / R[2]:
                continue
        # 计算平面参数
        #参考A(x-x1)+B(y-y1)+C(z-z1)=0,代入(x2,y2,z2)和(x3,y3.z3)
        a = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]);
        b = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]);
        c = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
        d = 0 - (a * p1[0] + b * p1[1] + c * p1[2]);

        dis = abs(a*X[:, 0] + b*X[:, 1]+ c*X[:, 2]+d) / (a**2 + b**2 + c** 2) ** 0.5
        idset = []
        for i, d in enumerate(dis):
            if d < tao:
                idset.append(i)
        # 再使用最小二乘法,得到更加准确的a,b,c,d
        p = PlaneLeastSquare(X[idset])
        a, b, c, d = p[0], p[1], -1, p[2]
        dic[len(idset)] = [a, b, c, d]
        if len(idset) > t * (1 - e):
            break
        count += 1

    parm = dic[max(dic.keys())]
    a, b, c, d = parm
    dis = abs(a*X[:, 0] + b*X[:, 1]+ c*X[:, 2]+d) / (a**2 + b**2 + c** 2) ** 0.5
    idset = []
    odset = []
    for i, d in enumerate(dis):
        if d < tao:
            idset.append(i)
        else:
            odset.append(i)
    print("地面的内点数，外点数:",len(idset),len(odset))
    return np.array(idset),np.array(odset),a,b,c,d

#点云旋转--显示
def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 10.0)
    return False

if __name__ == '__main__':
    ##原始数据的加载
    #读取数据
    origin_points = read_velodyne_bin("000000.bin")
    #建立点云对象
    pcd=open3d.geometry.PointCloud()
    #加载点云数据
    pcd.points=open3d.utility.Vector3dVector(origin_points)
    #改变点云颜色
    c=[0,0,0]
    cs=np.tile(c,(origin_points.shape[0],1))
    pcd.colors=open3d.utility.Vector3dVector(cs)
    #添加坐标系
    FOR1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    #显示点云
    open3d.visualization.draw_geometries([FOR1,pcd])
    print("原始数据的个数、维数:",origin_points.shape[0],origin_points.shape[1])

    ##通过RANSAC算法获得点云的地面
    planeids,othersids,pa,pb,pc,pd=PlaneRANSAC(origin_points,0.35)
    planedata = origin_points[planeids]
    planepcd = open3d.geometry.PointCloud()
    planepcd.points = open3d.utility.Vector3dVector(planedata)
    c = [0, 0, 255]
    cs = np.tile(c, (planedata.shape[0], 1))
    planepcd.colors = open3d.utility.Vector3dVector(cs)

    otherdata = origin_points[othersids]
    otherpcd = open3d.geometry.PointCloud()
    otherpcd.points = open3d.utility.Vector3dVector(otherdata)
    c = [255, 0, 0]
    cs = np.tile(c, (otherdata.shape[0], 1))
    otherpcd.colors = open3d.utility.Vector3dVector(cs)
    open3d.visualization.draw_geometries([planepcd, otherpcd])
    print("地面方程为a,b,c,d:",pa[0],pb[0],pc,pd)

    ##sklearn库下的DBSCAN进行聚类
    Css = sklearn.cluster.DBSCAN(eps=0.50, min_samples=4).fit(otherdata)
    ypred = np.array(Css.labels_)
    print("聚类的个数:",ypred.shape[0])
    ddraw = []
    colorset = [[222, 0, 0], [0, 224, 0], [0, 255, 255], [222, 244, 0], [255, 0, 255], [128, 0, 0]]
    for cluuus in set(ypred):
        kaka = np.where(ypred == cluuus)
        ppk = open3d.geometry.PointCloud()
        ppk.points = open3d.utility.Vector3dVector(otherdata[kaka])
        if len(ppk.points)<10:
            continue
        c = colorset[cluuus % 3]
        if cluuus == -1:
            c = [0, 0, 0]
        cs = np.tile(c, (otherdata[kaka].shape[0], 1))
        ppk.colors = open3d.utility.Vector3dVector(cs)
        ddraw.append(ppk)
    ddraw.append(planepcd)

    open3d.visualization.draw_geometries(ddraw)
    #点云绕着轴旋转
    #open3d.visualization.draw_geometries_with_animation_callback(ddraw,rotate_view)