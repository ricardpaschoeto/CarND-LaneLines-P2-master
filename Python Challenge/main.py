# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:06:01 2020

@author: paschoeto
"""

from pipeline import Pipeline
import numpy as np

## Project and challenge video
src = np.float32([(600, 460), (280, 720), (1110, 720), (720, 460)])
dest = np.float32([(300, 0), (300, 720), (960, 720), (960, 0)])

p = Pipeline(src=src,dest=dest)



#p.imageProcessing()
p.videoClip('../challenge_video.mp4', 'challenge_video.mp4')

    
