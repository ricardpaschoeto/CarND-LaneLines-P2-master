# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:14:03 2020

@author: paschoeto
"""
from moviepy.editor import VideoFileClip
from camera_calibrating import Camera
from color_gradient import ColorGradient
from undistorted_transform import Perspective
from lane_fit import Lane
import pipeline as p
import cv2
import numpy as np

toCompute = False
camera = Camera()
color = ColorGradient(gamma=0.5)
perspective = Perspective()
lane = Lane()

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result, left_curverad, right_curverad, offset = p.pipeline(image, camera, color, perspective, lane, toCompute)
    insertVideoValues(result, left_curverad, right_curverad, offset)
    
    return result

def video_clip(path):
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    
    white_output = 'test_videos_output/' + path
    clip1 = VideoFileClip('../project_video.mp4').subclip(0,50)
    #clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    # fl_image(lambda image: change_image(image, myparam))
    #left_line_params, right_line_params = None, None
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    
def insertVideoValues(img, left_curverad, right_curverad, offset):
    position = (10,50)
    cv2.putText(img,"Offset: " + str(np.round(offset,2)),position,cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)
    position = (10,90)
    cv2.putText(img,"Left Curve Radius(km): " + str(np.round(left_curverad/1000,3)),position,cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)
    position = (10,130)
    cv2.putText(img,"Right Curve Radius(km): " + str(np.round(right_curverad/1000,3)),position,cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)

video_clip('project_video.mp4')