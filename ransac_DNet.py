import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
import random


from sklearn import linear_model



def SVD(points):
	# 二维，三维均适用
	# 二维直线，三维平面
	pts = points.copy()
	# 奇异值分解
	c = np.mean(pts, axis=0)
	A = pts - c # shift the points
	A = A.T #3*n
	u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True) # A=u*s*vh
	normal = u[:,-1]

	# 法向量归一化
	nlen = np.sqrt(np.dot(normal,normal))
	normal = normal / nlen
	# normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
	# u 每一列是一个方向
	# s 是对应的特征值
	# c >>> 点的中心
	# normal >>> 拟合的方向向量
	return u,s,c,normal


class plane_model(object):
	def __init__(self):
		self.parameters = None

	def calc_inliers(self,points,dst_threshold):
		c = self.parameters[0:3]
		n = self.parameters[3:6]
		dst = abs(np.dot(points-c,n))
		ind = dst<dst_threshold
		return ind

	def estimate_parameters(self,pts):
		num = pts.shape[0]
		if num == 3:
			c = np.mean(pts,axis=0)
			l1 = pts[1]-pts[0]
			l2 = pts[2]-pts[0]
			n = np.cross(l1,l2)
			scale = [n[i]**2 for i in range(n.shape[0])]
			#print(scale)
			n = n/np.sqrt(np.sum(scale))
		else:
			_,_,c,n = SVD(pts)

		params = np.hstack((c.reshape(1,-1),n.reshape(1,-1)))[0,:]
		self.parameters = params
		return params

	def set_parameters(self,parameters):
		self.parameters = parameters


def ransac_planefit(points, ransac_n, max_dst, stop_inliers_ratio, max_trials=1000, initial_inliers=None):
	# RANSAC 平面拟合
	pts = points.copy()
	num = pts.shape[0]
	cc = np.mean(pts,axis=0)
	iter_max = max_trials
	best_inliers_ratio = 0 #符合拟合模型的数据的比例
	best_plane_params = None
	best_inliers = None
	best_remains = None
	for i in range(iter_max):
		sample_index = random.sample(range(num),ransac_n)
		sample_points = pts[sample_index,:]
		plane = plane_model()
		plane_params = plane.estimate_parameters(sample_points)
		#  计算内点 外点
		index = plane.calc_inliers(points,max_dst)
		inliers_ratio = pts[index].shape[0]/num

		if inliers_ratio > best_inliers_ratio:
			best_inliers_ratio = inliers_ratio
			best_plane_params = plane_params
			bset_inliers = pts[index]
			bset_remains = pts[index==False]

		if best_inliers_ratio > stop_inliers_ratio:
			# 检查是否达到最大的比例
			# print("iter: %d\n" % i)
			# print("best_inliers_ratio: %f\n" % best_inliers_ratio)
			break

	return best_plane_params,bset_inliers,bset_remains


class RANSAC_filter:
	def __init__(self):
		self.parameters = None
 
	def read_txt(self, path):
		"""
		read txt file into a np.ndarray.
		
		Input：
		------
		path: file path
		row_skip: skip the first rows to read data
		split_char: spliting character
		num_range: data range of each number
		Output：
		------
		data: data read. data is np.array([]) when reading error happened
						data is np.array([]) when nan or NaN appears
						data is np.array([]) when any number is out of range
		"""
		# 筛选cam_point中的零元素 
		cam_point = path
		cam_filter_res = []
		for dta in cam_point:
			if dta[0] == 0 and dta[1] == 0 and dta[2] == 0:
				continue
			cam_filter_res.append(dta)

		return 	np.array(cam_filter_res)

	def ransac_plane_detection(self, points,
								ransac_n, 
								max_dst, 
								stop_inliers_ratio=0.95, 
								max_trials=1000, 
								initial_inliers=None, 
								out_layer_inliers_threshold= 100, #230, 
								out_layer_remains_threshold= 100): #230):
		
		# inliers_num = out_layer_inliers_threshold + 1
		# remains_num = out_layer_remains_threshold + 1

		# plane_set = []
		# plane_inliers_set = []
		# plane_inliers_num_set = []

		# data_remains = np.copy(points)

		# i = 0

		# while inliers_num>out_layer_inliers_threshold : #and remains_num>out_layer_remains_threshold
		# 	# robustly fit line only using inlier data with RANSAC algorithm
		# 	best_plane_params,pts_inliers,pts_outliers = ransac_planefit(data_remains, ransac_n, max_dst, max_trials=max_trials, stop_inliers_ratio=stop_inliers_ratio)

		# 	inliers_num = pts_inliers.shape[0]
		# 	remains_num = pts_outliers.shape[0]

		# 	if inliers_num>out_layer_inliers_threshold:
		# 		plane_set.append(best_plane_params)
		# 		plane_inliers_set.append(pts_inliers)
		# 		plane_inliers_num_set.append(inliers_num)
		# 		i = i+1
		# 			# print('------------> %d <--------------' % i)
		# 			# print(best_plane_params)

		# 		data_remains = pts_outliers

		# # sorting
		# plane_set = [x for _, x in sorted(zip(plane_inliers_num_set,plane_set), key=lambda s: s[0], reverse=True)]
		# plane_inliers_set = [x for _, x in sorted(zip(plane_inliers_num_set,plane_inliers_set), key=lambda s: s[0], reverse=True)]
		normal, point = self.ransac_plane_new(points, 1000)

		return normal, point#plane_set, plane_inliers_set #, data_remains
		
	def ransac_plane_new(self, points, iterations=100, threshold=0.01):
	

		# 创建一个RANSAC回归器
		ransac = linear_model.RANSACRegressor(max_trials= 100, residual_threshold = 0.1, min_samples = 0.2)

		# 将数据拟合到RANSAC回归器
		ransac.fit(points[:, [0, 1]], points[:, 2])

		# 获取平面方程的系数
		a, b = ransac.estimator_.coef_
		c = -1
		d = ransac.estimator_.intercept_

		# 计算平面的法向量
		normal = np.array([a, b, c])
		normal = normal / np.linalg.norm(normal)

		# 计算平面上的一个点
		point = np.array([0, 0, d])

		return normal, point
			
			

	def show_3dpoints(self, pointcluster,s=None,colors=None,quiver=None,q_length=10,tri_face_index=None):
		# pointcluster should be a list of numpy ndarray
		# This functions would show a list of pint cloud in different colors
		n = len(pointcluster)
		if colors is None:
			colors = ['r','g','b','c','m','y','k','tomato','gold']
			if n < 10:
				colors = np.array(colors[0:n])
			else: 
				colors = np.random.rand(n,3)
			
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')

		if s is None:
			s = np.ones(n)*2

		for i in range(n):
			ax.scatter(pointcluster[i][:,0],pointcluster[i][:,1],pointcluster[i][:,2],s=s[i],c=[colors[i]],alpha=0.6)

		if not (quiver is None):
			c1 = [random.random() for _ in range(len(quiver))]
			c2 = [random.random() for _ in range(len(quiver))]
			c3 = [random.random() for _ in range(len(quiver))]
			c = []
			for i in range(len(quiver)):
				c.append((c1[i],c2[i],c3[i]))
			cp = []
			for i in range(len(quiver)):
				cp.append(c[i])
				cp.append(c[i])
			c = c + cp
			ax.quiver(quiver[:,0],quiver[:,1],quiver[:,2],quiver[:,3],quiver[:,4],quiver[:,5],length=q_length,arrow_length_ratio=.2,pivot='tail',normalize=False,color=c)
		
		if not (tri_face_index is None):
			for i in range(len(tri_face_index)):
				for j in range(tri_face_index[i].shape[0]):
					index = tri_face_index[i][j].tolist()
					index = index + [index[0]]
					ax.plot(*zip(*pointcluster[i][index]))

		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')

		#ax.set_ylim([-20,60])

		plt.show()

		return 0


if __name__ == "__main__":
	path = "./out_cam_point_461.npy"
	# pcd = read_txt(path)#, row_skip=1, split_char=' ')
	# pcd = pcd[:,:3]
    # # 第三个参数：maxdst = 0.5
	# point_size = len(pcd)
	# plane_set, plane_inliers_set, data_remains = ransac_plane_detection(pcd, 3, 0.1, 0.95 max_trials=1000,  initial_inliers=None,
	# 								out_layer_inliers_threshold=point_size*0.1, out_layer_remains_threshold=point_size*0.9)
	# plane_set = np.array(plane_set)

	# print("================= 平面参数 ====================")
	# print(plane_set) #（cx,cy,cz,nx,ny,nz)
	# # 绘图 平面模型参数（平面上任意一点cX,cy,cX)+(平面法向量nx,ny,nz)
	# show_3dpoints(plane_inliers_set)
	# print("cam_point:{}".format(path))
	# print("================= 远点到平面的距离 ====================")
	# x0, y0, z0, A, B, C = plane_set[0]
	# D = -(A*(x0) + B*(y0) + C*(z0))
	# dist = abs(D)/math.sqrt(A**2 + B**2 + C**2)
	# # dist = x0*A + y0*B + z0*C 
	# print("相机坐标系下远点到平面的距离：{}".format(dist))
	# print("over!!!")
