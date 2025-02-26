import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
import draw_flow_hor

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
            # print("flow:",flow.shape)
            # print(flow)
            # print("flow_x:[0][0][0]:",flow[0][0][0])
            # print("flow_y:[0][0][1]:",flow[0][0][1])
            flow_x = flow[...,0]  # 每个像素点的x位移
            # print("flow_y:",flow_y.shape)
            # print("flow_y:",flow_y[0][0])


            move_distance_hor = 0


            for i in range(flow_x.shape[0]):
                for j in range(flow_x.shape[1]):
                    move_distance_hor = move_distance_hor + flow_x[i][j]

            flow_hor.append(move_distance_hor/pixs)

            print(flow_hor)
            print(crop_pic)

            prvs = next
            frame_num = frame_num + 1

            # draw_flow_hor.draw_flow_hor(flow_hor, fps * 3, 1616 // 29)
    return flow_hor

def compute_acc_flow(pos, fps):
    print("len(pos):",len(pos))
    print(pos)

    acc_flow = []
    acc_flow.append(pos[0])
    for i in range(1,len(pos)):
        if i % fps != fps-1:
            acc_flow1 = pos[i] + acc_flow[i-1]
            acc_flow.append(acc_flow1)
        else:
            acc_flow.append(pos[i])

    print("累加后的x光流:",acc_flow)
    print("累加后的x光流的个数:",len(acc_flow))
    return acc_flow











