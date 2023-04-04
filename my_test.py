# import cv2

# video_file = "samples/fall-09-cam0-1.mp4" # 동영상 파일 경로

# cap = cv2.VideoCapture(video_file) # 동영상 캡쳐 객체 생성  ---①
# frame = 0
# if cap.isOpened():                 # 캡쳐 객체 초기화 확인
#     while True:
#         ret, img = cap.read()      # 다음 프레임 읽기      --- ②
#         if ret:                 # 프레임 읽기 정상
#             cv2.imshow(video_file, img) # 화면에 표시  --- ③
#             print(frame)
#             frame += 1
#             k = cv2.waitKey() 
#             if k == ord('s'):
#                 break
#       # 25ms 지연(40fps로 가정)   --- ④
#         else:                       # 다음 프레임 읽을 수 없슴,
#             print('break')
#             break                   # 재생 완료
# else:
#     print("can't open video.")      # 캡쳐 객체 초기화 실패
# cap.release()                       # 캡쳐 자원 반납
# cv2.destroyAllWindows()

# import torch

# a = None
# file = open("result_txt/detected_inroi.txt", 'a')
# file.write(str(a))
# file.close()

# import cv2
# frame = cv2.imread('HAR-UP-frame/HAR-UP-1-2/frame2.png', cv2.IMREAD_UNCHANGED)
# cv2.imwrite('HAR-UP-frame/HAR-UP-1-3/frame2.png',frame)
# cv2.imshow('frame', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 파일 이름바꾸는 라인
# import os 
# activity = 11
# dir_path = f'HAR-UP-{activity}_video'
# r = 1
# for  i in range(1,18):
#     for j in range(1,4):
#         for k in range(1,3):
#             original_name = os.path.join(dir_path, f'Subject{i}Activity{activity}Trial{j}Camera{k}.avi')
#             new_name = os.path.join(dir_path, f'HAR-UP-{i}-{activity}-{j}-{k}.avi')
#             os.rename(original_name, new_name)
# #파일 이름 바꾸기


# print(os.path.join(dir_path,f'result{r}.t'))
# temp = 0
# sum=0
# for i in range(1000):
#     sum += i
# temp += 1
# print(sum)
# if temp < 5:
#     os.system('python testtest.py')


import os
import cv2
import time
import torch
import argparse
# import torch
# import numpy as np
from CameraLoader import CamLoader, CamLoader_Q
from Detection.Utils import ResizePadding


inp_dets = 384
resize_fn = ResizePadding(inp_dets, inp_dets)
cam_dir_path = "HAR-UP_video"

def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
par.add_argument('-C', '--camera',  # required=True,  # default=2,
                    help='Source of camera or video file path.')

args = par.parse_args()
cam_source = args.camera
data_info_list = cam_source.split(sep='/')[-1]
data_info = data_info_list.split(sep='-')
subject = data_info[-4]
trial = data_info[-2]
activity = data_info[-3]
camera = data_info[-1]
camera = camera.split(sep='.')[0]

cam = CamLoader_Q(cam_source, queue_size=2000, preprocess=preproc).start()
f = 0
while cam.grabbed():
    frame = cam.getitem()
    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #frame저장
    detectedpath = f'./out/detection/2-HAR-UP-{subject}-{activity}-{trial}-{camera}/result{f}.t'
    if not(os.path.isfile(detectedpath)):
        detected = None
    else : 
        detected = torch.load(detectedpath)
    if detected is not None:
        detected = detected.numpy()
        #print(float(detected[0]),float(detected[1]),float(detected[2]),float(detected[3]))
        image = cv2.rectangle(image, (int(detected[0]), int(detected[1])), (int(detected[2]), int(detected[3])), (0, 255, 0), 1)
    image = cv2.putText(image, '%d' % (f),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    winname = 'test'
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 500, 1000)
    #cv2.imshow('frame', image)
    cv2.imshow(winname, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    f += 1
cam.stop()
cv2.destroyAllWindows()
#frame 추출후 ViTPose 파일로 이동
