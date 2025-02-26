# ---------------draw_flow_eye_model------------------------------------------------------------

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import xlwt


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    print(window)
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
    full_path =  r'..\acc_flow_eye_model_result' + '\\' + str(name[:-4]) + '.txt'
    file = open(full_path, 'a')
    file.write(msg)  # msg也就是下面的Hello world!
    file.write(" "+"\n")
    # file.close()


def peak_valleys(y_av,video_name):
    # 峰值检测并显示
    y_av = np.array(y_av)
    plt.plot(np.arange(len(y_av)), y_av)

    value1 = signal.argrelextrema(y_av, np.greater)
    value2 = signal.argrelextrema(-y_av, np.greater)
    # 打印峰值个数及对应坐标
    num1 = len(value1[0])
    num2 = len(value2[0])
    print(num1,num2)
    plt.plot(signal.argrelextrema(y_av, np.greater)[0], y_av[signal.argrelextrema(y_av, np.greater)], 'o')
    plt.plot(signal.argrelextrema(-y_av, np.greater)[0], y_av[signal.argrelextrema(-y_av, np.greater)], '+')

    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.title("Peak_Valleys detection")
    picture_name = r'..\paper_pic' + '\\' + str(video_name[:-4]) + '_5' + ".png"
    plt.savefig(picture_name)
    # plt.show()
    plt.close()

    return num1, num2

def findHighSpot_smooth(data, video_path, moving_average_fps, savgol_filter_fps, picture_result_name,video_times):

    (filepath, video_name) = os.path.split(video_path)

    xaxis = np.linspace(0, len(data),len(data))
    data = np.array(data)
    x = range(0, len(data))  # X轴数据

    # DFA去趋势
    # # data_detrended = signal.detrend(data)
    # data_detrended = signal.detrend(data, axis=-1, type='l', bp=600, overwrite_data=False)
    # # detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    # plt.plot(xaxis, data)
    # plt.plot(xaxis, data_detrended)
    # plt.title("Detrended_data")
    # plt.show()

    y_moving_average = moving_average(data,moving_average_fps)

    # smooth_curve平滑
    y_smooth_curve= smooth_curve(data)

    y_moving_smooth= smooth_curve(y_moving_average)
    y_smooth_moving= moving_average(y_smooth_curve,moving_average_fps)


    # DFA去趋势
    data_detrended = signal.detrend(y_smooth_moving,bp=600)
    # detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    # plt.plot(xaxis, data)


    plt.plot(xaxis, data,'orangered', label = "Original curve")
    plt.legend(loc='upper right')
    plt.title("Original curve")
    picture_name = r'..\paper_pic' + '\\' + str(video_name[:-4]) + '_1' + ".png"
    plt.savefig(picture_name)
    # plt.show()
    plt.close()

    # plt.plot(x, y_moving_average, 'b', label = "Moving_average")
    # plt.show()
    plt.plot(x, y_smooth_curve, 'green', label = "S-G filtered")
    plt.title("S-G filtered curve")
    plt.legend(loc='upper right')
    picture_name = r'..\paper_pic' + '\\' + str(video_name[:-4]) + '_2' + ".png"
    plt.savefig(picture_name)
    # plt.show()
    plt.close()

    # plt.plot(x, y_moving_smooth, 'grey', label = "moving_smooth")
    plt.plot(x, y_smooth_moving, 'blue', label = "Moving_ave filtered")
    plt.title("Moving_average filtered curve")
    plt.legend(loc='upper right')
    picture_name = r'..\paper_pic' + '\\' + str(video_name[:-4]) + '_3' + ".png"
    plt.savefig(picture_name)
    # plt.show()
    plt.close()

    plt.plot(x, y_smooth_moving, 'blue', label="Moving_ave filtered")
    plt.plot(xaxis, data_detrended,'red',label="Detrended curve")
    plt.title("Detrended curve")
    plt.legend(loc='upper right')
    picture_name = r'..\paper_pic' + '\\' + str(video_name[:-4]) + '_4' + ".png"
    plt.savefig(picture_name)
    # plt.show()
    plt.close()

    # plt.xlabel('Frame')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.title("Waveform")
    # picture_name = r'..\acc_flow_pictures_model' + '\\' + str(video_name[:-4]) + ".png"
    # plt.savefig(picture_name)
    # plt.show()
    # plt.close()


    # # 峰值检测并显示
    # num1_moving_average, num2_moving_average = peak_valleys(y_moving_average)
    # num1_smooth_curve, num2_smooth_curve =peak_valleys(y_smooth_curve)
    num1_moving_smooth, num2_moving_smooth = peak_valleys(y_moving_smooth,video_name)
    num1_smooth_moving, num2_smooth_moving = peak_valleys(y_smooth_moving,video_name)

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

    # num1_moving_average_savgol, num2_moving_average_savgol = peak_valleys(y_moving_average_savgol)
    # num1_smooth_curve_savgol, num2_smooth_curve_savgol = peak_valleys(y_smooth_curve_savgol)
    num1_moving_smooth_savgol, num2_moving_smooth_savgol = peak_valleys(y_moving_smooth_savgol,video_name)
    num1_smooth_moving_savgol, num2_smooth_moving_savgol = peak_valleys(y_smooth_moving_savgol,video_name)

    # # 最大值/最小值结果保存到文档中
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



if __name__ =="__main__":

    # 处理文件夹
    acc_flow_result_all = "..//acc_flow_result_h//"
    # acc_flow_result_all = "..//acc_flow_result_eye_model_h//"
    result_name_list = os.listdir(acc_flow_result_all)

    # 写入excel文件
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('eye_tracker_result', cell_overwrite_ok=True)
    col = ('视频名称', 'moving_smooth移动次数（最大值）', 'moving_smooth移动次数（最小值）','smooth_moving最大值','smooth_moving最小值', 'moving_smooth_savgol最大值','moving_smooth_savgol最小值',
           'num1_smooth_moving_savgol最大值','num2_smooth_moving_savgol最小值','moving_average_fps','savgol_filter_fps')
    for i1 in range(0, 11):
        sheet.write(0, i1, col[i1])

    for i in range(len(result_name_list)):
        result_path = os.path.join(acc_flow_result_all, result_name_list[i])

        # result_path = "..//acc_flow_result//zyh_h_lr_fast_34.txt"
        acc_flow = np.loadtxt(result_path, delimiter=',')

        # static 基线过滤
        under_sum = 0
        for index in range(1, len(acc_flow)):
            if acc_flow[index] < 0.1:
                under_sum = under_sum + 1
        if under_sum/len(acc_flow) >= 0.7:
            acc_flow = [0 for _ in range(len(acc_flow))]

        (filepath, video_name) = os.path.split(result_path)
        name1,name2,name3 = video_name.split("_",2)
        print(name1,name2,name3)

        # video_path = "..//video" + "//" + "eye_model" + "//" + str(video_name[:-4]) + ".MOV"

        video_path = "..//video" + "//" + str(name1) + "_" + str(name2) + "_trim" + "//" + str(video_name[:-4]) + ".mp4"
        camera = cv2.VideoCapture(video_path)
        length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(camera.get(cv2.CAP_PROP_FPS))
        video_times = length // fps

        # model fps//3 fps//15
        video_name, num1_moving_smooth, num2_moving_smooth, \
        num1_smooth_moving, num2_smooth_moving, \
        num1_moving_smooth_savgol, num2_moving_smooth_savgol, \
        num1_smooth_moving_savgol, num2_smooth_moving_savgol, \
        moving_average_fps, savgol_filter_fps =  findHighSpot_smooth(acc_flow,video_path,fps,fps, video_name,video_times)


        result = [video_name ,  num1_moving_smooth, num2_moving_smooth,\
           num1_smooth_moving, num2_smooth_moving ,\
           num1_moving_smooth_savgol, num2_moving_smooth_savgol,\
           num1_smooth_moving_savgol, num2_smooth_moving_savgol,\
           moving_average_fps , savgol_filter_fps]


        for j in range(0, 11):
            sheet.write(i + 1, j, str(result[j]))

        savepath = '..//eye_model_test.xls'
        book.save(savepath)

