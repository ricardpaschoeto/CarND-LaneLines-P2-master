# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:06:01 2020

@author: paschoeto
"""

from camera_calibrating import Camera
from color_gradient import ColorGradient
from undistorted_transform import Perspective
from lane_fit import Lane
import pipeline as p
from line import Line

import matplotlib.image as mpimg
import numpy as np
import glob
import cv2

images = glob.glob("../test_images/*.jpg")
toCompute = False
#vertices = np.array([[(150,img.shape[0]),(530, 420), (img.shape[1]-500, 420), (img.shape[1]-50,img.shape[0])]], dtype=np.int32)

camera = Camera()
color = ColorGradient(gamma=0.5)
perspective = Perspective()
lane = Lane()
left_line_lane = Line()
right_line_lane = Line()

def insertVideoValues(img, left_curverad, right_curverad, offset):
    position = (10,50)
    cv2.putText(img,"Offset: " + str(np.round(offset,2)),position,cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)
    position = (10,90)
    cv2.putText(img,"Left Curve Radius(km): " + str(np.round(left_curverad/1000,3)),position,cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)
    position = (10,130)
    cv2.putText(img,"Right Curve Radius(km): " + str(np.round(right_curverad/1000,3)),position,cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)


for idx, fname in enumerate(images):
    img = mpimg.imread(fname)
    result, left_curverad, right_curverad, offset = p.pipeline(img, camera, color, perspective, lane, left_line_lane, right_line_lane, toCompute)
    
    insertVideoValues(result, left_curverad, right_curverad, offset)
    cv2.imwrite("../output_images/output" + str(idx) + ".jpg", result)
    
