# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:06:01 2020

@author: paschoeto
"""

import pipeline as p
import matplotlib.image as mpimg
import glob
import cv2

images = glob.glob("../test_images/*.jpg")
for idx, fname in enumerate(images):
    image = mpimg.imread(fname)
    result =p.pipeline(image)
    cv2.imwrite("../output_images/output" + str(idx) + ".jpg", result)