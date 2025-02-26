import matplotlib.pyplot as plt
import pywt
import numpy as np

# Get data:
result_path = "..//acc_flow_result//zgh_h_rl_slow_18.txt"
ecg = np.loadtxt(result_path, delimiter=',')
# ecg = pywt.data.ecg()  # 生成心电信号
index = []
data = []
for i in range(len(ecg)-1):
    X = float(i)
    Y = float(ecg[i])
    index.append(X)
    data.append(Y)

# Create wavelet object and define parameters
w = pywt.Wavelet('db8')  # 选用Daubechies8小波
maxlev = pywt.dwt_max_level(len(data), w.dec_len)
print("maximum level is " + str(maxlev))
threshold = 0.4  # Threshold for filtering

# Decompose into wavelet components, to the level selected:
coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解


for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波

datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

mintime = 0
maxtime = mintime + len(data) + 1
# x_m = range(0, len(data)+1)  # X轴数据

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(index[mintime:maxtime], data[mintime:maxtime])
# plt.plot(x_m, data)
plt.xlabel('Frame')
plt.ylabel('Amplitude')
# plt.xlabel('time (s)')
# plt.ylabel('microvolts (uV)')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
# plt.plot(x_m, datarec)
plt.plot(index[mintime:maxtime], datarec[mintime:maxtime-1])
# plt.xlabel('time (s)')
# plt.ylabel('ampltitude (uV)')
plt.xlabel('Frame')
plt.ylabel('Amplitude')
plt.title("De-noised signal using wavelet techniques")

plt.tight_layout()
plt.show()
