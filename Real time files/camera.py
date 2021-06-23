import cv2
import os
import random
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import models

# from dataset import VocDetectorDataset
# from eval_voc import evaluate
from predict import predict_image
from resnet_yolo import resnet50
# from yolo_loss import YoloLoss
from config import VOC_CLASSES, COLORS

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = torch.device('cpu')

load_network_path = 'detector_on_coco_v7.pth'
pretrained = True

# use to load a previously trained network
if load_network_path is not None:
    print('Loading saved network from {}'.format(load_network_path))
    net = resnet50().to(device)
    net.load_state_dict(torch.load(load_network_path,map_location=torch.device('cpu')))
    print("Loaded the model")
else:
    print('Loaded the pre-trained Imagenet model')
    net = resnet50(pretrained=pretrained).to(device)
    

print("Before Eval")   
net.eval()
print("After eval")
# from final_yolo import net

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        # im_name = 'bottles_photo.jpg'
        # image = cv2.imread(os.path.join(file_root_test, im_name))
        # image = cv2.imread(os.path.join(im_name))
        # print(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Type of image before passing argument",type(image))
        result = predict_image(net, image)

        for left_up, right_bottom, class_name, _, prob in result:
            color = COLORS[VOC_CLASSES.index(class_name)]
            cv2.rectangle(image, left_up, right_bottom, color, 2)
            label = class_name + str(round(prob, 2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                        color, -1)
            cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()