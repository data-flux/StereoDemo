#!/usr/bin/env python3
#
# Suggested use:
#   from StereoMatch import stereoMatch, calculatePointCloud, plotPointCloud
#
import cv2
import numpy as np
import math
try:
    import pptk
except:
    pass



def stereoMatch(img):
    numDisparities= 220 # Maximum disparity we want to detect (closest point)
    minDisparities= 20 # Minimum disparity we want to detect (furthest point)
    blockSize = 7 # matching window size
    P1_constant = 100 # should be 0 because we are not looking at flat objects
    P2_constant = 10_000 #non-zero for smoothing
    uniquenessRatio = 10 #low = noisy, high = fewer matches

    stereo = cv2.StereoSGBM_create(
        numDisparities=numDisparities, 
        blockSize=blockSize,
        minDisparity=minDisparities,
        P1 = P1_constant,
        P2 = P2_constant,
        disp12MaxDiff = 0,
        preFilterCap = 0,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = 0,
        speckleRange = 0,
        mode = cv2.StereoSGBM_MODE_HH
    )
    disparity = stereo.compute(*img).astype(np.float32)/16.0
    disparity[np.logical_or(disparity<minDisparities,disparity>numDisparities)] = np.nan
    return disparity

def calculatePointCloud(disparity):
    dims = (1920,1200)
    scaling = 1000. #mm/m
    focalmm =  16. #mm
    sensormm = 6.6 #mm
    baseline = 29. #mm
    focalpix = (focalmm / sensormm) * dims[0] #px
    Zfactor = baseline * focalpix / scaling #m*px
    centerX = dims[0]/2.0 #px
    centerY = dims[1]/2.0 #px


    Z = Zfactor / disparity #m
    Z[Z==np.inf]=np.nan
    mask = ~np.isnan(Z.flatten())

    Y,X = np.mgrid[0:dims[1],0:dims[0]]
    vertices = np.array([X.flatten(),Y.flatten(),Z.flatten()],dtype=np.float64)

    for i in range(vertices.shape[1]):
        vertices[0,i] = (vertices[0,i]-centerX) * vertices[2,i] / focalpix #m
        vertices[1,i] = (vertices[1,i]-centerY) * vertices[2,i] / focalpix #m

    vertices[[1,2],:] = -vertices[[1,2],:] #flip on Z and Y axes for better viewing
    return vertices,mask

def plotPointCloud(vertices,color=None,lookat=[0.,0.,0.],theta=math.pi/2):
    v = pptk.viewer(vertices.transpose())
    if color is not None:
        if type(color) is not tuple:
            color = (color,)
        v.attributes(*color)
    else:
        v.attributes([[1.0,1.0,1.0,0.05]])
    v.set(
        point_size=0.2/1000,
        #show_grid=False, 
        #show_axis=False, 
        r=1.0,
        bg_color=[0.,0.,0.,.1],
        lookat=lookat,
        phi = -math.pi/2,
        theta = theta,
        selected = [],
    )
    return v

