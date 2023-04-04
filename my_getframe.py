import os
import cv2
import time
# import torch
# import numpy as np
from CameraLoader import CamLoader, CamLoader_Q
from Detection.Utils import ResizePadding

activity = 11
inp_dets = 384
resize_fn = ResizePadding(inp_dets, inp_dets)
cam_dir_path = f"HAR-UP-{activity}_video"
def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

for subject in range(1,18):
    for trial in range(1,4):
        for camera in range(1,3):
            try:
                cam_path = f"HAR-UP-{subject}-{activity}-{trial}-{camera}.avi"
                cam_source = os.path.join(cam_dir_path, cam_path)
                cam = CamLoader_Q(cam_source, queue_size=2000, preprocess=preproc).start()
                f = 0
                while cam.grabbed():
                    frame = cam.getitem()
                    image = frame.copy()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #frame저장
                    frame_dir = f'HAR-UP-{activity}-frame/HAR-UP-{subject}-{activity}-{trial}-{camera}'
                    frame_name = f'frame{f}.png'
                    os.makedirs(frame_dir, exist_ok=True)
                    frame_path = os.path.join(frame_dir, frame_name)
                    cv2.imwrite(frame_path, image)
                    cv2.imshow('frame', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    f += 1
                # Clear resource.
                cam.stop()
                cv2.destroyAllWindows()
                #frame 추출후 ViTPose 파일로 이동
            except:
                pass
            