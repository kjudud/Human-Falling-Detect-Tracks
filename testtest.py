import os
import cv2
import time
import torch
import argparse
import numpy as np

detectedpath = './out/detection/2-HAR-UP-1-1-1-1/result3.t'          
detected = torch.load(detectedpath)

keypoints_path = './out/kpt/2-HAR-UP-1-1-1-1/result3.npy'
keypoints_data = np.load(keypoints_path)

print(detected.size())
print(keypoints_data.shape)
print(keypoints_data)
