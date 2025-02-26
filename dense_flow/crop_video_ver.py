import dlib
import cv2

# eye_range 左上方的点(x,y)，高width，宽height
def crop(video_path,eye_range):
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
        cv2.rectangle(frame, (eye_range[0], eye_range[1]), (eye_range[2], eye_range[3]), (0, 255, 0), 1)
        # 保存脸部图片

        img1 = frame[eye_range[1]:eye_range[3],eye_range[0]:eye_range[2]]
        cv2.imwrite(r"../video/vids/ver/sleeping/" + str((j + 1)) + '.png', img1)

        j = j + 1
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    return length,fps



