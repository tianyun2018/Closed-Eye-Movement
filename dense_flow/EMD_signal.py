# import matplotlib.pyplot as plt
# import pywt
# import numpy as np
# import os
# import numpy as np
# from scipy.signal import medfilt
# import scipy.io as io
# from scipy import signal
#
# result_path = "..//acc_flow_result//llw_h_lr_medium_18.txt"
# ecg = np.loadtxt(result_path, delimiter=',')
#
# # loadData=np.load('/mnt/data/acf/Bdata/Normal.npy',allow_pickle=True)#读取数据
# # ecg=loadData[0]
# # ecg1=ecg[:,1]#取通道II
#
# def normalize(data):
#     data = data.astype('float')
#     mx = np.max(data, axis=0).astype(np.float64)
#     mn = np.min(data, axis=0).astype(np.float64)
#     # Workaround to solve the problem of ZeroDivisionError
#     return np.true_divide(data - mn, mx - mn, out=np.zeros_like(data - mn), where=(mx - mn)!=0)
# ecg1=normalize(ecg)#归一化处理
# t1=int(0.2*500)
# t2=int(0.6*500)
# ecg2=medfilt(ecg1,t1+1)
# ecg3=medfilt(ecg2,t2+1)#分别用200ms和600ms的中值滤波得到基线
# ecg4=ecg1-ecg3#得到基线滤波的结果
#
#
# plt.plot(ecg1)#输出原图像
# plt.plot(ecg3)#输出基线轮廓
# plt.plot(ecg4)#基线滤波结果
# plt.show()


import matplotlib.pyplot as plt
import pywt
import numpy as np
import os
import numpy as np
from scipy.signal import medfilt
import scipy.io as io
from scipy import signal
# loadData=np.load('/mnt/data/acf/Bdata/Normal.npy',allow_pickle=True)#读取数据
# ecg=loadData[0]
# ecg1=ecg[:,1]#取通道II

result_path = "..//acc_flow_result//llw_h_lr_medium_18.txt"
ecg1 = np.loadtxt(result_path, delimiter=',')

#0.67Hz高通滤波
b = signal.firwin(51,0.1,pass_zero=False,fs=500)    # FIR高通滤波
ecg3 = signal.lfilter(b, 1, ecg1)

plt.plot(ecg1)
plt.plot(ecg3)
plt.show()
