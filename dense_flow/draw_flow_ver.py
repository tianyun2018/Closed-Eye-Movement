import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pylab as pl
import scipy.signal as signal
from scipy.interpolate import make_interp_spline
import os
import xlwt


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    # print(window)
    re = np.convolve(interval, window, 'same')
    return re


def smooth_curve(points, factor = 0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# 创建一个txt文件，文件名为mytxtfile,并向文件写入msg
def txt_create(name, msg):
    full_path =  r'..\acc_flow_time_and_frame_4_result' + '\\' + str(name[:-4]) + '.txt'
    file = open(full_path, 'a')
    file.write(msg)  # msg也就是下面的Hello world!
    file.write(" "+"\n")
    # file.close()

def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x

def peak_valleys(y_av):
    # 峰值检测并显示
    y_av = np.array(y_av)
    plt.plot(np.arange(len(y_av)), y_av)

    value1 = signal.argrelextrema(y_av, np.greater)
    value2 = signal.argrelextrema(-y_av, np.greater)
    # 打印峰值个数及对应坐标
    num1 = len(value1[0])
    num2 = len(value2[0])
    # print(num1,num2)
    plt.plot(signal.argrelextrema(y_av, np.greater)[0], y_av[signal.argrelextrema(y_av, np.greater)], 'o')
    plt.plot(signal.argrelextrema(-y_av, np.greater)[0], y_av[signal.argrelextrema(-y_av, np.greater)], '+')

    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.title("Peak_Valleys detection")
    # plt.show()
    plt.close()

    return num1, num2

def findHighSpot_smooth(data, video_path, moving_average_fps, savgol_filter_fps, video_times):
    xaxis = np.linspace(0, video_times ,len(data))
    plt.plot(xaxis, data)
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.title("Original_flow")
    plt.show()

    (filepath, video_name) = os.path.split(video_path)

    xaxis = np.linspace(0, len(data),len(data))
    data = np.array(data)
    x = range(0, len(data))  # X轴数据

    # # DFA去趋势
    # data_detrended = signal.detrend(data)
    # # detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    # plt.plot(xaxis, data)
    # plt.plot(xaxis, data_detrended)
    # plt.title("Detrended_data")
    # plt.show()

    # moving_average 平滑后
    # moving_average_fps = fps

    # static 处理 归一化

    # for index in range(1,len(data)):
    #     if data[index] < 0.1:
    #         data[index] = 0
    # data = MaxMinNormalization(data, np.max(data), np.min(data))

    y_moving_average = moving_average(data,moving_average_fps)

    # smooth_curve平滑
    y_smooth_curve= smooth_curve(data)

    y_moving_smooth= smooth_curve(y_moving_average)
    y_smooth_moving= moving_average(y_smooth_curve,moving_average_fps)

    # # MSE误差 -- fps
    # mse_e1 = sum((y_av1-data)**2)
    # mse_e2 = sum((y_av2-data)**2)
    # print(mse_e1,mse_e2,mse_e1-mse_e2)
    #
    # if mse_e1 >= mse_e2:
    #     y_av = y_av2
    #     moving_average_fps = fps //2
    # else:
    #     y_av = y_av1
    #     moving_average_fps = fps


    plt.plot(xaxis, data,'r', label = "Original_data")
    plt.plot(x, y_moving_average, 'b', label = "Moving_average")
    plt.plot(x, y_smooth_curve, 'g', label = "smooth_curve")
    plt.plot(x, y_moving_smooth, 'grey', label = "moving_smooth")
    plt.plot(x, y_smooth_moving, 'black', label = "smooth_moving")
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Original_data_and_Moving_average")
    # picture_name = r'..\acc_flow_pictures_1' + '\\' + str(video_name[:-4]) + ".png"
    # plt.savefig(picture_name)
    # plt.show()
    plt.close()


    # # 峰值检测并显示
    # plt.plot(np.arange(len(y_av)), y_av)
    # plt.plot(signal.argrelextrema(y_av, np.greater)[0], y_av[signal.argrelextrema(y_av, np.greater)], 'o')
    # plt.plot(signal.argrelextrema(-y_av, np.greater)[0], y_av[signal.argrelextrema(-y_av, np.greater)], '+')
    # plt.xlabel('Frame')
    # plt.ylabel('Amplitude')
    # plt.title("Peak_Valleys detection")
    # plt.show()
    num1_moving_average, num2_moving_average = peak_valleys(y_moving_average)
    num1_smooth_curve, num2_smooth_curve =peak_valleys(y_smooth_curve)
    num1_moving_smooth, num2_moving_smooth = peak_valleys(y_moving_smooth)
    num1_smooth_moving, num2_smooth_moving = peak_valleys(y_smooth_moving)

    # savgol_filter_fps
    # savgol_filter 平滑处理
    if (savgol_filter_fps % 2) == 0:
        savgol_filter_fps_odd = savgol_filter_fps + 1
    else:
        savgol_filter_fps_odd = savgol_filter_fps

    y_moving_average_savgol = signal.savgol_filter(y_moving_average, savgol_filter_fps_odd, 2)
    y_smooth_curve_savgol = signal.savgol_filter(y_smooth_curve, savgol_filter_fps_odd, 2)
    y_moving_smooth_savgol = signal.savgol_filter(y_moving_smooth, savgol_filter_fps_odd, 2)
    y_smooth_moving_savgol = signal.savgol_filter(y_smooth_moving, savgol_filter_fps_odd, 2)

    # plt.plot(xaxis, data,'r', label = "Original_data")
    # plt.plot(x, y_av, 'b', label = "Moving_average")
    # plt.plot(x, tmp_smooth1, label="Savgol_filter", color='green', linewidth=2)
    # plt.xlabel('Frame')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.title("Savgol_filter")
    # plt.show()


    # # 平滑后峰值检测并显示
    # plt.plot(np.arange(len(tmp_smooth1)), tmp_smooth1,'g')
    # value1 = signal.argrelextrema(tmp_smooth1, np.greater)
    # value2 = signal.argrelextrema(-tmp_smooth1, np.greater)
    # # 打印峰值个数及对应坐标
    # num1 = len(value1[0])
    # num2 = len(value2[0])

    num1_moving_average_savgol, num2_moving_average_savgol = peak_valleys(y_moving_average_savgol)
    num1_smooth_curve_savgol, num2_smooth_curve_savgol = peak_valleys(y_smooth_curve_savgol)
    num1_moving_smooth_savgol, num2_moving_smooth_savgol = peak_valleys(y_moving_smooth_savgol)
    num1_smooth_moving_savgol, num2_smooth_moving_savgol = peak_valleys(y_smooth_moving_savgol)

    # # 结果保存到文档中
    # txt_create(video_name, video_path)
    # txt_create(video_name, "moving_average_fps:" + str(moving_average_fps))
    # txt_create(video_name, "savgol_filter_fps:" + str(savgol_filter_fps))
    # txt_create(video_name, "检测眼动左右移动次数(最大值)：" + str(num1))
    # txt_create(video_name, "检测眼动左右移动次数(最小值)：" + str(num2))
    # txt_create(video_name, "检测眼动位移最大位置帧数(最大值)：" + str(value1))
    # txt_create(video_name, "检测眼动位移最大位置帧数(最小值)：" + str(value2))
    #
    # print(video_path)
    # print("moving_average_fps:", moving_average_fps)
    # print("检测眼动左右移动次数(最大值)：", num1)
    # print("检测眼动左右移动次数(最小值)：", num2)
    # print("检测眼动位移最大位置帧数(最大值)：", value1)
    # print("检测眼动位移最大位置帧数(最小值)：", value2)
    # # print(signal.argrelextrema(tmp_smooth1, np.greater))
    # # np.savetxt("..//acc_flow_result//" + video_name[:-4] + '.txt', acc_flow)
    #
    # plt.plot(signal.argrelextrema(tmp_smooth1, np.greater)[0],
    #          tmp_smooth1[signal.argrelextrema(tmp_smooth1, np.greater)], 'o')
    # plt.plot(signal.argrelextrema(-tmp_smooth1, np.greater)[0],
    #          tmp_smooth1[signal.argrelextrema(-tmp_smooth1, np.greater)], '+')
    # plt.title("Dense_Flow left_or_right")
    # plt.xlabel('Frame')
    # plt.ylabel('Amplitude')
    # picture_result_name = r'..\acc_flow_pictures_1' + '\\' + str(video_name[:-4]) + "_result.png"
    # # plt.savefig(picture_result_name)
    # # plt.show()
    # plt.close()
    return video_name ,  num1_moving_smooth, num2_moving_smooth,\
           num1_smooth_moving, num2_smooth_moving ,\
           num1_moving_smooth_savgol, num2_moving_smooth_savgol,\
           num1_smooth_moving_savgol, num2_smooth_moving_savgol,\
           moving_average_fps , savgol_filter_fps

# def findHighSpot_sleeping_smooth(data,fps,video_times,threshold_high,threshold_low):
#
#     xaxis = np.linspace(0, video_times ,len(data))
#     plt.plot(xaxis, data)
#     plt.xlabel('Frame')
#     plt.ylabel('Amplitude')
#     plt.title("Original_flow")
#     plt.show()
#
#     data = np.array(data)
#     x = range(0, len(data))  # X轴数据
#
#     y_av = moving_average(data, 10)
#     plt.plot(x, y_av, 'b')
#     plt.xlabel('Frame')
#     plt.ylabel('Amplitude')
#     plt.title("Moving_average")
#     plt.show()
#
#     # 峰值检测并显示
#     plt.plot(np.arange(len(y_av)), y_av)
#     # plt.plot(signal.argrelextrema(y_av, np.greater)[0], y_av[signal.argrelextrema(y_av, np.greater)], 'o')
#     # plt.plot(signal.argrelextrema(-y_av, np.greater)[0], y_av[signal.argrelextrema(-y_av, np.greater)], '+')
#     # plt.show()
#
#     # 局部值处理
#     x1_plot = signal.argrelextrema(y_av, np.greater)[0]
#     j1 = 0
#     for i in signal.argrelextrema(y_av, np.greater)[0]:
#         # if abs(y_av[i]) <= threshold_high:
#         if y_av[i] <= threshold_high and y_av[i] >= threshold_low:
#             x1_plot[j1] = 0
#         j1 = j1 + 1
#     x11_plot=[]
#     for i in x1_plot:
#         if i !=0:
#             x11_plot.append(i)
#
#     x2_plot = signal.argrelextrema(-y_av, np.greater)[0]
#     j2=0
#     for i in signal.argrelextrema(-y_av, np.greater)[0]:
#         if y_av[i] <= threshold_high and y_av[i] >= threshold_low:
#             x2_plot[j2] = 0
#         j2 = j2 + 1
#     x22_plot=[]
#     for i in x2_plot:
#         if i !=0:
#             x22_plot.append(i)
#
#     plt.plot(x11_plot, y_av[x11_plot], 'o')
#     plt.plot(x22_plot, y_av[x22_plot], '+')
#     plt.xlabel('Frame')
#     plt.ylabel('Amplitude')
#     plt.title("Peak_Valleys detection")
#     plt.show()
#
#     # 平滑处理
#     if (fps % 2) == 0:
#         fps_odd = fps + 1
#     else:
#         fps_odd = fps
#
#     tmp_smooth1 = signal.savgol_filter(y_av, fps_odd, 3)
#     plt.plot(x, tmp_smooth1, label="data_ff", color='green', linewidth=2)
#     plt.xlabel('Frame')
#     plt.ylabel('Amplitude')
#     plt.title("Savgol_filter")
#     plt.show()
#
#     # DFA去趋势
#     tmp_smooth1_detrended = signal.detrend(tmp_smooth1)
#     # detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
#     plt.plot(xaxis, tmp_smooth1)
#     plt.plot(xaxis, tmp_smooth1_detrended)
#     plt.title("Detrended_data")
#     plt.show()
#
#     # 平滑后峰值检测并显示
#     plt.plot(np.arange(len(tmp_smooth1)), tmp_smooth1)
#     value = signal.argrelextrema(tmp_smooth1, np.greater)
#
#     # 局部值处理
#     u1_plot = value[0]
#     j1 = 0
#     for i in value[0]:
#         if y_av[i] <= threshold_high and y_av[i] >= threshold_low:
#             u1_plot[j1] = 0
#         j1 = j1 + 1
#     u11_plot=[]
#     for i in u1_plot:
#         if i !=0:
#             u11_plot.append(i)
#
#     u2_plot = signal.argrelextrema(-tmp_smooth1, np.greater)[0]
#     j2=0
#     for i in signal.argrelextrema(-tmp_smooth1, np.greater)[0]:
#         if y_av[i] <= threshold_high and y_av[i] >= threshold_low:
#             u2_plot[j2]=0
#         j2 = j2 + 1
#     u22_plot=[]
#     for i in u2_plot:
#         if i !=0:
#             u22_plot.append(i)
#
#     # 打印峰值个数及对应坐标
#     num1 = len(u11_plot)
#     num2 = len(u22_plot)
#     print("检测眼动左右移动次数(最大值)：", num1)
#     print("检测眼动左右移动次数(最小值)：", num2)
#     print("检测眼动位移最大位置帧数(最大值)：", u11_plot)
#     print("检测眼动位移最大位置帧数(最小值)：", u22_plot)
#
#     plt.plot(u11_plot, tmp_smooth1[u11_plot], 'o')
#     plt.plot(u22_plot, tmp_smooth1[u22_plot], '*')
#
#
#     # print(signal.argrelextrema(tmp_smooth1, np.greater))
#     # plt.plot(signal.argrelextrema(tmp_smooth1, np.greater)[0],
#     #          tmp_smooth1[signal.argrelextrema(tmp_smooth1, np.greater)], 'o')
#     # plt.plot(signal.argrelextrema(-tmp_smooth1, np.greater)[0],
#     #          tmp_smooth1[signal.argrelextrema(-tmp_smooth1, np.greater)], '+')
#     plt.title("Dense_Flow left_or_right")
#     plt.xlabel('Frame')
#     plt.ylabel('Amplitude')
#     plt.show()

# def draw_flow_hor(pos,fps,video_times):
#     print("len(pos):",len(pos))
#     print(pos)
#
#     acc_flow = []
#     acc_flow.append(pos[0])
#     acc = 0
#     for i in range(1,len(pos)):
#         if i % fps != fps-1:
#             acc_flow1 = pos[i] + acc_flow[i-1]
#
#             acc_flow.append(acc_flow1)
#         else:
#             acc_flow.append(pos[i])
#
#     print("平均后的x:",acc_flow)
#     print("平均后的x的个数:",len(acc_flow))
#     return acc_flow
#
#     # xaxis = np.linspace(0, video_times ,len(acc_flow))
#     # plt.plot(xaxis, acc_flow)
#     #
#     # plt.title("Dense_Flow left_or_right")
#     # plt.xlabel('Time')
#     # plt.ylabel('Amplitude')
#     # plt.show()
#
#
#     # plt.draw()
#     # plt.pause(1)
#     # plt.close()


# 4-1
# def draw_flow_hor(pos,fps,video_times):
#     print("len(pos):",len(pos))
#     print(pos)
#
#     acc_flow = []
#     final_flow = []
#     # acc_flow.append(pos[0])
#     acc = 0
#     for i in range(0,len(pos)-1):
#         if i == 0:
#             final_flow.append(pos[0])
#         else:
#             acc_flow1 = pos[i] + final_flow[i-1]
#             if abs(acc_flow1 - pos[0])<= 1e-2:
#                 final_flow.append(pos[i+1])
#             else:
#                 final_flow.append(acc_flow1)
#
#     print("累加后的x:",final_flow)
#     print("累加后的x的个数:",len(final_flow))
# #
#     xaxis = np.linspace(0, video_times ,len(final_flow))
#     plt.plot(xaxis, final_flow)
#     plt.title("Dense_Flow left_or_right")
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.show()


# just_acc
# def draw_flow_hor_abs(pos,fps,video_times):
#     print("len(pos):",len(pos))
#
#     acc_flow = []
#     acc_flow.append(pos[0])
#     acc = 0
#     for i in range(1,len(pos)):
#         acc_flow1 = pos[i] + acc_flow[i-1]
#         acc_flow.append(acc_flow1)
#
#     print("平均后的x:",acc_flow)
#     print("平均后的x的个数:",len(acc_flow))
#
#     xaxis = np.linspace(0, video_times ,len(acc_flow))
#     plt.plot(xaxis, acc_flow)
#     plt.title("Dense_Flow left_or_right")
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.show()
#     # plt.draw()
#     # plt.pause(0.1)
#     # plt.close()
#     return acc_flow
#
# def draw_flow_ver(pos,fps,video_times):
#     print("len(pos):",len(pos))
#     print(pos)
#
#     acc_flow = []
#     acc_flow.append(pos[0])
#     acc = 0
#     for i in range(1,len(pos)):
#         # if i % fps != fps-1:
#         acc_flow1 = pos[i] + acc_flow[i-1]
#
#         acc_flow.append(acc_flow1)
#         # else:
#         #     acc_flow.append(pos[i])
#
#     print("平均后的x:",acc_flow)
#     print("平均后的x的个数:",len(acc_flow))
#
#     xaxis = np.linspace(0, video_times ,len(acc_flow))
#     plt.plot(xaxis, acc_flow)
#
#     plt.title("Dense_Flow left_or_right")
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.show()
#     # plt.draw()
#     # plt.pause(1)
#     # plt.close()


if __name__ =="__main__":

    # 处理文件夹

    acc_flow_result_all = "..//acc_flow_result_h//"
    result_name_list = os.listdir(acc_flow_result_all)

    # 写入excel文件
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('eye_tracker_result', cell_overwrite_ok=True)
    col = ('视频名称', 'moving_smooth移动次数（最大值）', 'moving_smooth移动次数（最小值）','smooth_moving最大值','smooth_moving最小值', 'moving_smooth_savgol最大值','moving_smooth_savgol最小值',
           'num1_smooth_moving_savgol最大值','num2_smooth_moving_savgol最小值','moving_average_fps','savgol_filter_fps')
    for i1 in range(0, 11):
        sheet.write(0, i1, col[i1])

    err_num = 0

    for i in range(len(result_name_list)):
        result_path = os.path.join(acc_flow_result_all, result_name_list[i])

        # result_path = "..//acc_flow_result//zyh_h_lr_fast_34.txt"
        acc_flow = np.loadtxt(result_path, delimiter=',')


        # # static 基线过滤
        # # ver 0.1 0.7
        # under_sum = 0
        # for index in range(1, len(acc_flow)):
        #     if acc_flow[index] < 0.1:
        #         under_sum = under_sum + 1
        # if under_sum/len(acc_flow) >= 0.8:
        #     acc_flow = [0 for _ in range(len(acc_flow))]


        (filepath, video_name) = os.path.split(result_path)
        name1,name2,name3 = video_name.split("_",2)
        # print(video_name.split("_"))
        if(video_name.split("_")[-1] == "static.txt"):
            video_eye_move_count = 0
        else:
            video_eye_move_count = video_name.split("_")[-1][:-4]
            # video_eye_move_count = name5[:-4]

        # print(video_eye_move_count)
        video_path = "..//video" + "//" + str(name1) +"_" + str(name2) + "_trim" + "//" + str(video_name[:-4]) + ".mp4"
        camera = cv2.VideoCapture(video_path)
        length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(camera.get(cv2.CAP_PROP_FPS))
        video_times = length // fps


        video_name, num1_moving_smooth, num2_moving_smooth, \
        num1_smooth_moving, num2_smooth_moving, \
        num1_moving_smooth_savgol, num2_moving_smooth_savgol, \
        num1_smooth_moving_savgol, num2_smooth_moving_savgol, \
        moving_average_fps, savgol_filter_fps =  findHighSpot_smooth(acc_flow,video_path,fps,fps//11, video_times)
        # video_path_name ,num_max, num_min, moving_average_fps = findHighSpot_smooth(acc_flow,fps//2,video_path,video_times)

        result = [video_name ,  num1_moving_smooth, num2_moving_smooth,\
           num1_smooth_moving, num2_smooth_moving ,\
           num1_moving_smooth_savgol, num2_moving_smooth_savgol,\
           num1_smooth_moving_savgol, num2_smooth_moving_savgol,\
           moving_average_fps , savgol_filter_fps]


        if (abs(int(num2_moving_smooth_savgol) - int(video_eye_move_count)) > 3 ):
            print(video_name, num2_moving_smooth_savgol)
            err_num = err_num + 1



        for j in range(0, 11):
            sheet.write(i + 1, j, str(result[j]))

        savepath = '..//eog_result_h_1.xls'
        book.save(savepath)

    print(err_num)


    # # 处理单个视频
    #
    #     # result_path = "..//acc_flow_result//zyh_h_lr_fast_34.txt"
    #     result_path = "..//acc_flow_result//llw_h_lr_medium_18.txt"
    #     acc_flow = np.loadtxt(result_path, delimiter=',')
    #     # print(acc_flow)
    #     (filepath, video_name) = os.path.split(result_path)
    #     name1, name2, name3 = video_name.split("_", 2)
    #     print(name1, name2, name3)
    #     video_path = "..//video" + "//" + str(name1) + "_" + str(name2) + "_trim" + "//" + str(video_name[:-4]) + ".mp4"
    #     # video_path = "..//video//zgh_h" + "//" + str(video_name[:-4]) + ".mp4"
    #     camera = cv2.VideoCapture(video_path)
    #     length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    #     fps = int(camera.get(cv2.CAP_PROP_FPS))
    #     video_times = length // fps
    #
    #     # print(np.median(acc_flow))
    #     # print(np.mean(acc_flow))
    #     # print(np.max(acc_flow))
    #     # print(np.min(acc_flow))
    #     # threshold_high = np.max(acc_flow) / 4
    #     # threshold_low = np.min(acc_flow) / 3
    #     # print(threshold_low)
    #     # print(threshold_high)
    #
    #     # findHighSpot_smooth(acc_flow, fps, video_path, video_times)
    #     findHighSpot_smooth(acc_flow, 30, video_path, video_times)
    #     # findHighSpot_sleeping_smooth(acc_flow, fps, video_times,threshold_high,threshold_low)
    #
    #     # print(np.median(acc_flow))
    #     # print(np.mean(acc_flow))