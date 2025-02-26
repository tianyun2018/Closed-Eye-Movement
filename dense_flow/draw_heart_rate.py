import numpy as np
import cv2
import matplotlib.pyplot as plt

# x=[64,64]
# y = [1209,0]
# x1 = [62,62]
# y1 = [323,0]
# x2 = [58,58]
# y2 = [92,0]

plt.axis([0, 100, 0, 1500])
plt.axvline(54.26,ymin = 0,ymax =58)
plt.axvline(57.34,ymin = 0,ymax =68)
plt.axvline(58.16,ymin = 0,ymax =92)
plt.axvline(58.74,ymin = 0,ymax =128)
plt.axvline(60.59,ymin = 0,ymax =158)
plt.axvline(60.59,ymin = 0,ymax =198)
plt.axvline(61.59,ymin = 0,ymax =358)
plt.axvline(62.31,ymin = 0,ymax =558)
plt.axvline(64.27,ymin = 0,ymax =1258)


# plt.plot(x, y, color = 'blue')
# plt.plot(x1,y1, color = 'blue')
# plt.plot(x2,y2, color = 'blue')

plt.title("Average_Heart_Rate")
plt.xlabel('Heart_Rate')
plt.ylabel('Hz')
plt.show()


