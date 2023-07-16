from pyexiv2 import Image
import math

EARTH_RADIUS = 6371.0*10e2
PI = 3.14



def calcuate_dis(p1,p2):

    p1['lat'] = p1['lat']  * PI / 180
    p2['lat'] = p2['lat']  * PI / 180
    p1['lon'] = p1['lon']  * PI / 180
    p2['lon'] = p2['lon']  * PI / 180
 
    dis = math.cos(p2['lat']) * math.cos(p1['lat']) * math.cos(p2['lon'] -p1['lon']) + math.sin(p1['lat']) * math.sin(p2['lat'])
    
    distance = EARTH_RADIUS * math.acos(dis)

    return distance



def caculate_dph(i,j,h):
    x_dis = h*math.tan(72.18/5472*(i-2735)*PI/180)
    y_dis = h*math.tan(51.8/3648*(1824-j)*PI/180)

    dis = math.sqrt(x_dis**2+y_dis**2+h**2)
    return x_dis,y_dis,dis

img = Image(r"./wy_utils/0000_gt.jpg")

xmp_info = img.read_xmp()
exif_info = img.read_exif()

cam_intrinsics = xmp_info['Xmp.drone-dji.DewarpData'].split(';')[1].split(',')

fx,fy = float(cam_intrinsics[0]),float(cam_intrinsics[1]) 
cx,cy = float(cam_intrinsics[2]),float(cam_intrinsics[3])

# TODO 读取无人机经纬高信息以及相机位姿、无人机位姿信息。
uav_lon = float(xmp_info['Xmp.drone-dji.GpsLongtitude'])
uav_lat = float(xmp_info['Xmp.drone-dji.GpsLatitude'])
uav_rAlt = float(xmp_info['Xmp.drone-dji.RelativeAltitude'])

gbl_roll = float(xmp_info['Xmp.drone-dji.GimbalRollDegree'])
gbl_yaw = float(xmp_info['Xmp.drone-dji.GimbalYawDegree'])
gbl_pitch = float(xmp_info['Xmp.drone-dji.GimbalPitchDegree'])

uav_xSped = float(xmp_info['Xmp.drone-dji.FlightXSpeed'])
uav_ySped = float(xmp_info['Xmp.drone-dji.FlightYSpeed'])
uav_zSped = float(xmp_info['Xmp.drone-dji.FlightZSpeed'])


# TODO 输入目标经纬度信息 计算两点距离[gt]
p1,p2 = dict(),dict()

#------------------------------------
# 修改为目标lon lat 
p2['lon'] = 117.03848949 #car1
p2['lat'] = 39.13313236



# 修改为目标在图像中的位置
center_x = 524
center_y = 344
#------------------------------------

p1['lon'] = uav_lon
p1['lat'] = uav_lat
print(f'无人机经纬度坐标：lon:{uav_lon},lat:{uav_lat}')

distance = math.sqrt(calcuate_dis(p1,p2)**2 + uav_rAlt**2) # 根据点云数据得到的距离结果
print(f'点云数据测得的目标距离：{distance}')


# TODO 根据输入目标bbox 得到预测目标距离


# x_dis,y_dis,dis = caculate_dph(center_x,center_y,uav_rAlt)

# print(f'目标距离：x方向-->{x_dis},y方向↑：{y_dis},直线距离：{dis}')

