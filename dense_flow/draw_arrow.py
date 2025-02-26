import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math


def dense(path, fps,pixs):
    pics = os.listdir(path)
    pics.sort(key=lambda x:int(x[:-4]))

    frame1 = cv2.imread(os.path.join(path, pics[0]))
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    flow_hor = []
    frame_num = 1
    for crop_pic in pics:
        if crop_pic !='1.png':
            frame2 = cv2.imread(os.path.join(path, crop_pic))
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # 返回一个两通道的光流向量，实际上是每个点的像素位移值
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            flow_x = flow[...,0]  # 每个像素点的x位移
            flow_y = flow[...,1]  # 每个像素点的x位移

            # print("flow_x:",flow_x)
            # print("x_max:",np.max(flow_x))
            # print("x_min:",np.min(flow_x))

            move_distance_hor = 0

            for i in range(flow_x.shape[0]):
                for j in range(flow_x.shape[1]):
                    move_distance_hor = move_distance_hor + flow_x[i][j]

                    # print("i:",frame_num,i,j,flow_x[i][j])

            # flow_x_original = flow_x + flow_x0

            for i in range(60,flow_x.shape[0]-50,50):
                for j in range(100,flow_x.shape[1]-30,50):
                    # draw_arrow
                    # if abs(flow_x[i][j]>=0.01):
                    # print("frame:", frame1.shape)
                    # print("flow_x:", flow_x.shape)
                    startx = i
                    starty = j
                    endx = int(i + flow_y[i][j] * 5)
                    # endy = int(j + flow_x[i][j] * 20)
                    # print(flow_x[i][j])
                    # cv2.arrowedLine(frame2, (starty, startx), (endy, endx), (255, 0, 0), 2, 8, 0, 0.3)  # 画箭头


                    if(abs(flow_x[i][j]) > 1):
                        endy = int(j + flow_x[i][j] * 5)
                    # elif(abs(flow_x[i][j]) > 0.8):
                    #     endy = int(j + flow_x[i][j] * 20)
                    elif(abs(flow_x[i][j]) > 0.5):
                        endy = int(j + flow_x[i][j] * 20)
                    elif(abs(flow_x[i][j]) > 0.2):
                        endy = int(j + flow_x[i][j] * 50)
                    elif(abs(flow_x[i][j]) > 0.01):
                        endy = int(j + flow_x[i][j] * 200)
                    else:
                        endy = int(j + flow_x[i][j] * 3000)
                    # if(flow_x[i][j] > 0):
                    #     endy = int(j+10)
                    #     endx = int(i)
                    # if(flow_x[i][j] < 0):
                    #     endy = int(j-10)
                    #     endx = int(i)
                    # cv2.arrowedLine(frame2, (starty, startx), (endy, endx), (255, 0, 0), 2, 9, 0, 0.3)  # 画箭头
                    cv2.arrowedLine(frame2, (starty, startx), (endy, endx), (255, 0, 0), 2, 8, 0, 0.3)  # 画箭头
                    # cv2.circle(frame2, (starty, startx), 2,(255, 0, 0),3)  # 画箭头
                    # cv2.circle(frame, (min_x, mid_y), 2, (0, 0, 255), 3)

            cv2.imshow('frame2', frame2)
            k = cv2.waitKey(30) & 0xff
            if k == 27 & k == 0xff:
                break

            flow_hor.append(move_distance_hor/pixs)

            print(flow_hor)
            print(crop_pic)



            prvs = next
            frame_num = frame_num + 1

            # draw_flow_hor.draw_flow_hor(flow_hor, fps * 3, 1616 // 29)
    return flow_hor

def compute_acc_flow(pos,fps):
    print("len(pos):",len(pos))
    print(pos)

    acc_flow = []
    acc_flow.append(pos[0])
    acc = 0
    for i in range(1,len(pos)):
        if i % fps != fps-1:
            acc_flow1 = pos[i] + acc_flow[i-1]

            acc_flow.append(acc_flow1)
        else:
            acc_flow.append(pos[i])

    print("累加后的x光流:",acc_flow)
    print("累加后的x光流的个数:",len(acc_flow))
    return acc_flow


def crop(video_path, eye_range):

    camera = cv2.VideoCapture(video_path)
    length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(camera.get(cv2.CAP_PROP_FPS))

    if not camera.isOpened():
        print("cannot open camera")
        exit(0)
    j = 0

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        # cv2.rectangle(frame, (300, 250), (60, 180), (0, 255, 0), 1)
        cv2.rectangle(frame, (eye_range[0], eye_range[1]), (eye_range[2], eye_range[3]), (0, 255, 0), 1)

        # 保存脸部图片

        # img1 = frame[180:250, 60:300]
        img1 = frame[eye_range[1]:eye_range[3], eye_range[0]:eye_range[2]]

        (filepath, vids_name) = os.path.split(video_path)
        vids_dir = r'..\video' + '\\' + 'draw_arrow' + '\\' + str(vids_name[:-4])

        if not os.path.exists(vids_dir):
            os.makedirs(vids_dir)

        cv2.imwrite(vids_dir +'\\'+ str((j + 1)) + '.png', img1)

        j = j + 1

        # cv2.resizeWindow('Camera', 200, 400)  # 自己设定窗口图片的大小
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    return length,fps,vids_dir

if __name__ =="__main__":
    # video_path =  "..//video//zyh_h_trim//zyh_h_lr_slow_17.mp4"
    # video_path =  "..//video//zgh_h_trim//zgh_h_rl_medium_26.mp4"
    # video_path =  r"E:\eye\eye_tracker\video\eye_model\model_h_lr_11_18.MOV"
    video_path = "..//video//wd_h_trim//wd_h_rl_fast_52.mp4"
    eye_range = [520, 380, 1400, 600]  # wd 7h
    # eye_range = [450,350,1450,600]   # zyh 1h/v

    # eye_range = [700, 250, 1500, 500]  # zgh 2h
    # eye_range = [950, 450, 1350, 550]  # eye_model
    # eye_range = [520, 380, 1350, 600]  # pjh 8h

    pixs = (eye_range[2]-eye_range[0])*(eye_range[3]-eye_range[1])
    print("pixs:",pixs)

    # length, fps, vids_dir = crop(video_path, eye_range)


    (fpath, vids_name) = os.path.split(video_path)
    # vids_dir = r'..\video' + '\\' + 'draw_arrow' + '\\' + str(vids_name[:-4])
    # (filepath, vids_name) = os.path.split(video_path)
    vids_dir = r'..\video' + '\\' + 'vids' + '\\' + str(vids_name[:-4])
    camera = cv2.VideoCapture(video_path)
    length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(camera.get(cv2.CAP_PROP_FPS))

    print("帧数length:",length)
    print("帧率fps:",fps)


    # flow_hor = dense_move_hor.dense(vids_dir,fps,pixs)
    flow_ver = dense(vids_dir,fps,pixs)

    # video_times = length // fps

    # acc_flow = dense_move_hor.compute_acc_flow(flow_hor,fps)
    # acc_flow = compute_acc_flow(flow_ver,fps)

    # (filepath, video_name) = os.path.split(video_path)
    # np.savetxt("..//acc_flow_result_v//"+ video_name[:-4] +'.txt' , acc_flow)








