import os
import cv2
import time
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

#source = '../Data/test_video/test7.mp4'
#source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
source = '../Data/falldata/Home/Videos/video (1).avi'
#source = 2


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

def IoU(box1, box2):
    # obtain x1, y1, x2, y2 of the intersection
    box1 = box1.numpy()
    box2 = box2.numpy()
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)


    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=2000, preprocess=preproc).start()
        # subject, trial, camera정보
        data_info_list = cam_source.split(sep='/')[-1]
        data_info = data_info_list.split(sep='-')
        subject = data_info[-4]
        activity = data_info[-3]
        trial = data_info[-2]
        camera = data_info[-1]
        camera = camera.split(sep='.')[0]
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    #frame_size = cam.frame_size
    #scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'FMP4')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    #내가 추가한 라인
    r = 0
    first_detect = False
    det_tensor_save_dir = f'out/detection/2-HAR-UP-{subject}-{activity}-{trial}-{camera}'
    os.makedirs(det_tensor_save_dir, exist_ok=True)
    #정확도 테스트하려고 action저장할 리스트
    action_list = []
    action_win = 10
    decision = 0
    previous_detected = torch.tensor([])
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()

        # Detect humans bbox in the frame with detector model.
        #내가 주석으로 바꾼 라인
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        tensor_save_state = False
        min_iou = 0
        if detected is not None:
            # camera 1
            if camera == '1':
                if not first_detect: 
                    for det_idx in range(detected.shape[0]):
                        if (first_detect == False) and (detected[det_idx][3] > 300) and \
                            ((detected[det_idx][2] - detected[det_idx][0] + detected[det_idx][3] - detected[det_idx][1]) >= 150):
                            tensorpath = os.path.join(det_tensor_save_dir,f'result{r}.t')
                            torch.save(detected[det_idx], tensorpath)
                            previous_detected = detected[det_idx]
                            first_detect = True
            #camera 2
            else :
                if not first_detect: 
                    for det_idx in range(detected.shape[0]):
                        if (first_detect == False) and (detected[det_idx][0]>=30 and detected[det_idx][2]<=230) and \
                            (detected[det_idx][2]-detected[det_idx][0]+detected[det_idx][3]-detected[det_idx][1]>150) and (detected[det_idx][3] >= 220):
                            tensorpath = os.path.join(det_tensor_save_dir,f'result{r}.t')
                            torch.save(detected[det_idx], tensorpath)
                            previous_detected = detected[det_idx]
                            first_detect = True
            # camera 1                
            if camera == '1':
                for track in tracker.tracks:
                    det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                    detected = torch.cat([detected, det], dim=0) if detected is not None else det
                    # camera1 : 바운딩 박스 topleft y가 130이상 bottomright x가 350이하이고 맨하탄 거리가 200이상이면 텐서 저장
                    for det_idx in range(detected.shape[0]):
                        if first_detect:
                            iou = IoU(previous_detected, detected[det_idx])
                            if detected[det_idx][4] != torch.tensor([0.5],dtype=torch.float32):
                                iou *= 1.5
                            if (iou > 0.3) and (detected[det_idx][3] > 300) and \
                            ((detected[det_idx][2] - detected[det_idx][0] + detected[det_idx][3] - detected[det_idx][1]) >= 150):
                                if iou > min_iou:
                                    min_iou = iou
                                    tensorpath = os.path.join(det_tensor_save_dir,f'result{r}.t')
                                    torch.save(detected[det_idx], tensorpath)
                                    tensor_save_state = True
                        if (not first_detect) and (detected[det_idx][3] > 280) and \
                            ((detected[det_idx][2] - detected[det_idx][0] + detected[det_idx][3] - detected[det_idx][1]) >= 150):
                            tensorpath = os.path.join(det_tensor_save_dir,f'result{r}.t')
                            torch.save(detected[det_idx], tensorpath)
                            tensor_save_state = True
                            first_detect = True
                    if tensor_save_state == True:
                        previous_detected = torch.load(tensorpath)
        # camera2 : 바운딩 박스 topleft,bottomright x가 80이상 280이하이고 bottomright y가 190이상이면 텐서 저장
        else :
            for track in tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det
                for det_idx in range(detected.shape[0]):
                    if first_detect:
                        iou = IoU(previous_detected, detected[det_idx])
                        if detected[det_idx][4] != torch.tensor([0.5],dtype=torch.float32):
                            iou *= 1.4
                        else :
                            iou *= 0.7
                        if (iou > 0.2) and (detected[det_idx][0]>=30 and detected[det_idx][2]<=230) and \
                            (detected[det_idx][2]-detected[det_idx][0]+detected[det_idx][3]-detected[det_idx][1]>150) and (detected[det_idx][3] >= 220):
                            if iou > min_iou:
                                min_iou = iou
                                tensorpath = os.path.join(det_tensor_save_dir,f'result{r}.t')
                                torch.save(detected[det_idx], tensorpath)
                                tensor_save_state = True
                    if (not first_detect) and (detected[det_idx][0]>=30 and detected[det_idx][2]<=230) and \
                        (detected[det_idx][2]-detected[det_idx][0]+detected[det_idx][3]-detected[det_idx][1]>150) and (detected[det_idx][3] >= 220):
                            tensorpath = os.path.join(det_tensor_save_dir,f'result{r}.t')
                            torch.save(detected[det_idx], tensorpath)
                            tensor_save_state = True
                            first_detect = True     
                if tensor_save_state == True:
                    previous_detected = torch.load(tensorpath) 


        detections = []  # List of Detections object for tracking.
        if detected is not None:
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])


            #Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                #내가 추가한 내용
                #여기까지 추가한 내용
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

                #정확도 체크 하는 라인
                # vote_num = []
                # if len(action_list) < i+1:
                #     action_list.append([action_name])
                # else:
                #     action_list[i].append(action_name)
                # #action_list.append(action_name)
                # if len(action_list[i]) == action_win:
                #     for class_name in action_model.class_names:
                #         vote_num.append(action_list[i].count(class_name))
                #     action_list[i].pop(0)
                #     #print(action_list)
                #     if (decision==0) and (vote_num.index(max(vote_num)) == 6):
                #         # file = open('./togetacc.txt', 'a')
                #         # file.write('1\n')
                #         # file.close()
                #         decision = 1

            

            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                   0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            writer.write(frame)
        winname = 'test'
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 500, 500)
        cv2.imshow(winname, frame)                
        #cv2.imshow('frame', frame)
        r += 1
        key = cv2.waitKey(1) 
        if key == ord('q'):
            break

    file = open("result_txt/framenum-7.txt", 'a')
    file.write(str(f))
    file.write(f'-{subject}-{activity}-{trial}-{camera}')
    file.write('\n')
    file.close()
    # Clear resource.
    cam.stop()
    if outvid:              
        writer.release()
    cv2.destroyAllWindows()
