import numpy as np
import cv2
import os
import glob

def save_frame_camera_key(dir_path, interval , ext='jpg', delay=1):
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    #fps1 = cap1.get(cv2.CAP_PROP_FPS)
    #fps2 = cap2.get(cv2.CAP_PROP_FPS)
    if not cap1.isOpened() or not cap2.isOpened():
        return
    #base_path = os.path.join(dir_path, basename)
    left_path = os.path.join(dir_path, "left_image/")
    right_path = os.path.join(dir_path, "right_image/")

    os.makedirs(left_path, exist_ok=True)
    os.makedirs(right_path, exist_ok=True)

    save = 0

    #print(fps1," frames/sec")
    #print(fps2," frames/sec")
    
    result = glob.glob(os.path.join(left_path,'*.jpg'))
    num=len(result)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 == False or ret2 == False:
            break
        cv2.imshow('Left', frame1)
        cv2.imshow('Right', frame2)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('s'):
            framecount = 0
            save = 1
        elif key == ord('c'):
            save = 0
        elif key == ord('q'):
            save = 0
            break
        if save == 1:
            framecount += 1
            if framecount % interval == 0:
                print("snapshot !!")
                cv2.imwrite('{}{:010}.{}'.format(left_path, num, ext), frame1)
                cv2.imwrite('{}{:010}.{}'.format(right_path, num, ext), frame2)
                num+=1
        else:
            continue

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

# sorted(glob.glob(os.path.join(dirPathPattern,'*.jpg')))

dirPathPattern = "snapshot/"

save_frame_camera_key(dirPathPattern, 30)
#To can't open above one camera:
#https://stackoverflow.com/questions/53888878/opencv-warn0-terminating-async-callback-when-attempting-to-take-a-picture?rq=1
# all_camera_idx_available = []

# for camera_idx in range(10):
#     cap = cv2.VideoCapture(camera_idx)
#     if cap.isOpened():
#         print(f'Camera index available: {camera_idx}')
#         all_camera_idx_available.append(camera_idx)
#         cap.release()
# while(True):
#     ret, frame = cap.read()

#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
