# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:10:15 2020

@author: paschoeto
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Perspective:
    
    def applyPerspective(self, img, mtx, dist):
        #region = self.region(img, vertices)
        warped, M = self.cornersUnwarp(img, mtx, dist)
        
        return warped, M
    
    def applyInversePerpectives(self,image, undist, warped, Minv, left_fitx, right_fitx, ploty):
        
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        return result

    def cornersUnwarp(self,img, mtx, dist):
        # Use the OpenCV undistort() function to remove distortion
        gray = cv2.undistort(img, mtx, dist, None, mtx)
        warped = None
        M = None
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])
    
        # For source points I'm grabbing the outer four detected corners
        src = np.float32([(580, 460), (205, 720), (1110, 720), (703, 460)])
        
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([(320, 0), (320, 720), (960, 720), (960, 0)])
        #[(150,img.shape[0]),(530, 420), (img.shape[1]-500, 420), (img.shape[1]-50,img.shape[0])]
        
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(gray, M, img_size)
    
        # Return the resulting image and matrix
        return warped, M
    
    def plotUndist(self, warped, left_curverad, right_curverad, offset):
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # f.tight_layout()
        # ax1.imshow(img)
        # ax1.set_title('Original Image', fontsize=50)
        plt.imshow(warped)
        plt.text(5, 50, str(left_curverad), style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        #plt.set_title('Undistorted and Warped Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
