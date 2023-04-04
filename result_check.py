# import os
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--file_dir",help='ex) voting_result/queue_10')
# args = parser.parse_args()

# acc_activity_save_dir = args.file_dir
# activities = [1,2,3,4,5,7,8,9,10,11]
# for activity in activities:
#     line_num = 0
#     #activity에 오타있음
#     voting_result_path_1 = os.path.join(acc_activity_save_dir, f'acc-avtivity-{activity}.txt')    
#     f = open(voting_result_path_1, 'r')
#     fall_num = 0
#     not_fall_num = 0

#     voting_result_path_2 = os.path.join(acc_activity_save_dir, f'acc-avtivity-{activity}-alpha.txt')    
#     g = open(voting_result_path_2, 'r')
#     fall_num = 0
#     not_fall_num = 0
#     fall_num2 = 0
#     not_fall_num2 = 0

#     while True:
#         line = f.readline()
#         line2 = g.readline()
#         if not line:
#             break
#         result = line[0]
#         result2 = line2[0]
#         if result =='1':
#             fall_num += 1
#         else:
#             not_fall_num += 1

#         if result2 =='1':
#             fall_num2 += 1
#         else:
#             not_fall_num2 += 1            
#         line_num += 1
#     print('fall:'+str(fall_num), 'normal:'+str(not_fall_num))
#     print('fall:'+str(fall_num2), 'normal:'+str(not_fall_num2))
#     # print(f'line_num : {line_num}')
#     print(f'activity : {activity}',end='\n\n')

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_dir",help='ex) voting_result/queue_10')
args = parser.parse_args()

acc_activity_save_dir = args.file_dir
activities = [1,2,3,4,5,7,8,9,10,11]
for activity in activities:
    line_num = 0
    #activity에 오타있음
    voting_result_path_1 = os.path.join(acc_activity_save_dir, f'acc-avtivity-{activity}.txt')    
    f = open(voting_result_path_1, 'r')
    fall_num = 0
    not_fall_num = 0

    while True:
        line = f.readline()
        if not line:
            break
        result = line[0]
        if result =='1':
            fall_num += 1
        else:
            not_fall_num += 1
        
        line_num += 1
    print('fall:'+str(fall_num), 'normal:'+str(not_fall_num))
    # print(f'line_num : {line_num}')
    print(f'activity : {activity}',end='\n\n')