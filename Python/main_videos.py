# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:14:03 2020

@author: paschoeto
"""
from moviepy.editor import VideoFileClip
import pipeline as p

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = p.pipeline(image)

    return result

def video_clip(path):
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    
    white_output = 'test_videos_output/' + path
    clip1 = VideoFileClip('../challenge_video.mp4').subclip(0,50)
    #clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    # fl_image(lambda image: change_image(image, myparam))
    #left_line_params, right_line_params = None, None
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    
video_clip('challenge_video.mp4')