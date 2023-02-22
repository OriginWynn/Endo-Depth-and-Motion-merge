import numpy as np
import cv2
import os
import glob

def save_frame_camera_key(device_num, dir_path, basename , interval , ext='jpg', delay=1, window_name='frame'):
    cap = cv2.VideoCapture(device_num)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)

    base_path = os.path.join(dir_path, basename)

    save = 0

    print(fps," frames/sec")
    while True:
        result = glob.glob(os.path.join(dir_path,'*.jpg'))
        ret, frame = cap.read()
        if ret == False :
            break
        cv2.imshow(window_name, frame)

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
                cv2.imwrite('{}_{:08}.{}'.format(base_path, len(result), ext), frame)
        else:
            continue

    cap.release()
    cv2.destroyWindow(window_name)

# sorted(glob.glob(os.path.join(dirPathPattern,'*.jpg')))

dirPathPattern = "snapshot/"

save_frame_camera_key(1, dirPathPattern, 'camera_snapshot', 30)

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