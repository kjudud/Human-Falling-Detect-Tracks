import os
import my_notify
import cv2

queue = [10,15,20,25,30]

for q in queue:
    for activity in range(1,6):
        for subject in range(1,18):
            for trial in range(1,4):
                for camera in range(1,3):
                    try:
                        command_line = f'python my_alpha_main_queuesize_bel.py -C HAR-UP-{activity}_video/HAR-UP-{subject}-{activity}-{trial}-{camera}.avi --queue_size {q} --note_acc True'
                        os.system(command_line)
                        print(f'{subject}-{activity}-{trial}-{camera}')
                    except:
                        pass

    for activity in range(7,12):
        for subject in range(1,18):
            for trial in range(1,4):
                for camera in range(1,3):
                    try:
                        command_line = f'python my_alpha_main_queuesize_bel.py -C HAR-UP-{activity}_video/HAR-UP-{subject}-{activity}-{trial}-{camera}.avi --queue_size {q} --note_acc True'
                        os.system(command_line)
                        print(f'{subject}-{activity}-{trial}-{camera}')                
                    except:
                        pass                   

# for q in queue:
#     for activity in range(1,6):
#         for subject in range(1,18):
#             for trial in range(1,4):
#                 for camera in range(1,3):
#                     try:
#                         command_line = f'python my_alpha_main_queuesize_max.py -C HAR-UP-{activity}_video/HAR-UP-{subject}-{activity}-{trial}-{camera}.avi --queue_size {q} --note_acc True'
#                         os.system(command_line)
#                         print(f'{subject}-{activity}-{trial}-{camera}')
#                     except:
#                         pass

#     for activity in range(7,12):
#         for subject in range(1,18):
#             for trial in range(1,4):
#                 for camera in range(1,3):
#                     try:
#                         command_line = f'python my_alpha_main_queuesize_max.py -C HAR-UP-{activity}_video/HAR-UP-{subject}-{activity}-{trial}-{camera}.avi --queue_size {q} --note_acc True'
#                         os.system(command_line)
#                         print(f'{subject}-{activity}-{trial}-{camera}')                
#                     except:
#                         pass   

my_notify.send_message()

# queue = [15,20,25,30]
# activities = [7,8,11]
# for q in queue:
#     for activity in activities:
#         for subject in range(1,18):
#             for trial in range(1,4):
#                 for camera in range(1,3):
#                     try:
#                         command_line = f'python my__main_queuesize_voting.py -C HAR-UP-{activity}_video/HAR-UP-{subject}-{activity}-{trial}-{camera}.avi --queue_size {q} --note_acc True'
#                         os.system(command_line)
#                         print(f'{subject}-{activity}-{trial}-{camera}')
#                     except:
#                             pass

# my_notify.send_message()