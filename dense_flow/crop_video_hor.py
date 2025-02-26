import dlib
import cv2
import os
# eye_range 左上方的点(x,y)，右下方的点
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
        vids_dir = r'..\video' + '\\' + 'vids' + '\\' + str(vids_name[:-4])

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



