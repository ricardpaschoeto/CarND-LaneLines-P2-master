# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:50:41 2020

@author: paschoeto
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from line import Line

class Lane:
    
    def __init__(self, nwindows=9, margin=100, minpix=50, ym_per_pix= 30/720, xm_per_pix = 3.7/700):
        # HYPERPARAMETERS
        self.ym_per_pix = ym_per_pix # meters per pixel in y dimension
        self.xm_per_pix = xm_per_pix # meters per pixel in x dimension
        # Choose the number of sliding windows
        self.nwindows = nwindows
        # Set the width of the windows +/- margin
        self.margin = margin
        # Set minimum number of pixels found to recenter window
        self.minpix = minpix
        # Fit coeficients
        self.left_fit = None
        self.right_fit = None
        self.ploty = None
        # Fit Real World coeficients
        self.left_fit_cr = np.empty((2))
        self.right_fit_cr = np.empty((2))
        # Data Left Lines
        self.left_line_lane = Line()
        self.right_line_lane = Line()
        
    def applyLaneCalculations(self, warped):
        
        # if (not self.left_line_lane.getDetected()):
        #     leftx, lefty, rightx, righty = self.findLane(warped)
            
        # leftx, lefty, rightx, righty = self.searchAroundPoly(warped)
        
        # try:
        #     left_fitx,leftx,lefty, right_fitx = self.fitPolynomial(warped, leftx, lefty, rightx, righty)
        # except TypeError:
        #     leftx, lefty, rightx, righty = self.findLane(warped)
        #     left_fitx,leftx,lefty, right_fitx = self.fitPolynomial(warped, leftx, lefty, rightx, righty)
        
        # Fit new polynomials
        left_fitx,leftx,lefty, right_fitx,rightx,righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = self.fitPolynomial(warped)
        self.left_line_lane.setRecentXfitted(left_fitx)
        self.right_line_lane.setRecentXfitted(right_fitx)
        #self.plotSearchAround(warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fitx,right_fitx)
       
        # # Determine the curvature of the lane and vehicle position with respect to center.
        left_curverad, right_curvera = self.measureCurvatureReal()
        self.left_line_lane.setRadius(left_curverad)
        self.right_line_lane.setRadius(right_curvera)
        #print((left_curverad,right_curvera))
        
        ## verify if paralell
        paralell = self.tangent()
        
        # # Determine center position and x distance
        center, xdistance, offset = self.vehicleCenterPos()
        self.left_line_lane.setLineBasePos(center)
        self.right_line_lane.setLineBasePos(center)
        
        return left_fitx, right_fitx, left_curverad, right_curvera, offset
        
    def setLeftFit(self, left_fit):
        self.left_fit = left_fit
        
    def getLeftFit(self):
        return self.left_fit
    
    def setRightFit(self, right_fit):
        self.right_fit = right_fit
        
    def getRightFit(self):
        return self.right_fit
    
    def setPloty(self, ploty):
        self.ploty = ploty
    
    def getPloty(self):
        return self.ploty
   
    def findLane(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        leftx, lefty, rightx, righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = self.windows(binary_warped, leftx_current, rightx_current)

        return leftx, lefty, rightx, righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds
        
    def windows(self, binary_warped, leftx_current, rightx_current):
       
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),[0,255,0], 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),[0,255,0], 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        self.left_line_lane.setAllx(leftx)
        self.left_line_lane.setAlly(lefty)
        
        self.right_line_lane.setAllx(rightx)
        self.right_line_lane.setAlly(righty)
        
  
        return leftx, lefty, rightx, righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds

    def searchAroundPoly(self, binary_warped):
        # HYPERPARAMETER
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_fit = self.left_line_lane.getBestFit()
        right_fit =  self.right_line_lane.getBestFit()
        
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - self.margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + self.margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - self.margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + self.margin)))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
   
        return leftx, lefty, rightx, righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds
    
    def fitPolynomial(self, binary_warped):
        
        leftx = lefty = rightx = righty = []
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        if(self.left_line_lane.getDetected() or self.right_line_lane.getDetected()):
            leftx, lefty, rightx, righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = self.searchAroundPoly(binary_warped)
        else:
            leftx, lefty, rightx, righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = self.findLane(binary_warped)
            self.left_line_lane.setDetected(True)
            self.right_line_lane.setDetected(True)
            
        try:
            # Fit a second order polynomial to each using `np.polyfit`
            self.setLeftFit(np.polyfit(lefty, leftx, 2))
            self.setRightFit(np.polyfit(righty, rightx, 2))            
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            leftx, lefty, rightx, righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = self.findLane(binary_warped)
            self.setLeftFit(np.polyfit(lefty, leftx, 2))
            self.setRightFit(np.polyfit(righty, rightx, 2)) 

        
        self.left_line_lane.setCurrentFit(self.getLeftFit())
        self.right_line_lane.setCurrentFit(self.getRightFit())
        
        self.left_line_lane.setDiffs(self.left_line_lane.getBestFit() - self.getLeftFit())
        self.right_line_lane.setDiffs(self.right_line_lane.getBestFit() - self.getRightFit())
        
        left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]

        return left_fitx,leftx,lefty, right_fitx,rightx,righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds

    def measureCurvatureReal(self):
        '''
        Calculates the curvature of polynomial functions in pixels.
        '''       
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)*self.ym_per_pix
    
        self.left_fit_cr[0] = self.left_fit[0]*(self.xm_per_pix/self.ym_per_pix**(2))
        self.left_fit_cr[1] = self.left_fit[1]*(self.xm_per_pix/self.ym_per_pix)
        
        self.right_fit_cr[0] = self.right_fit[0]*(self.xm_per_pix/self.ym_per_pix**(2))
        self.right_fit_cr[1] = self.right_fit[1]*(self.xm_per_pix/self.ym_per_pix)
        
        left_curverad = ((1 + (2*self.left_fit_cr[0]*y_eval + self.left_fit_cr[1])**(2))**(3/2))/np.abs(2*self.left_fit_cr[0])  ## Implement the calculation of the left line here
        right_curverad = ((1 + (2*self.right_fit_cr[0]*y_eval + self.right_fit_cr[1])**(2))**(3/2))/np.abs(2*self.right_fit_cr[0])  ## Implement the calculation of the right line here
        
        return left_curverad, right_curverad
    
    def vehicleCenterPos(self):
        xleft = self.left_line_lane.getBestX()*self.xm_per_pix
        xright = self.right_line_lane.getBestX()*self.xm_per_pix
        
        center = np.abs((xright - xleft)/2)
        xdistance = np.abs(xright - xleft)
        
        offset = 3.7 - xdistance
        
        return center, xdistance, offset
    
    def tangent(self):
        left_tan = 2*self.left_line_lane.getBestFit()[0]*self.left_line_lane.getAlly() + self.left_line_lane.getBestFit()[1]
        right_tan = 2*self.right_line_lane.getBestFit()[0]*self.right_line_lane.getAlly() + self.right_line_lane.getBestFit()[1]
        
        avg_left_angle = np.abs(180*np.arctan(1/np.average(left_tan))/np.pi)
        avg_right_angle = np.abs(180*np.arctan(1/np.average(right_tan))/np.pi)
        
        # print(avg_left_angle)
        # print(avg_right_angle)
        
        if(np.abs(avg_left_angle - avg_right_angle) <= 5.0):
            return True
        else:
            return False
        
    def plotLlines(self,binary_warped, left_fitx,leftx,lefty, right_fitx,rightx,righty):
        ## Visualization ##
        # Plots the left and right polynomials on the lane lines
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        
        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        plt.plot(left_fitx, self.ploty, color='yellow')
        plt.plot(right_fitx, self.ploty, color='yellow')
        plt.imshow(out_img)
        plt.show()

    def plotSearchAround(self, binary_warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fitx,right_fitx ):
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, 
                                  self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, 
                                  self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, self.ploty, color='yellow')
        plt.plot(right_fitx, self.ploty, color='yellow')
        ## End visualization steps ##
        
        plt.imshow(result)
        plt.show()

    
