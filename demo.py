#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import random
from funcy import lmap
from code.StereoMatch import stereoMatch, calculatePointCloud, plotPointCloud
from code.Filter import removeBackground, zBand, interactiveZBand
from code.Data import loadImageSet, listImageSets
from code.Surface import cylindricalCoordinates, ransacSinoidFit, flattenSurface

def demo(imset):
    print(f"[*] Loading images...")
    imageSet = loadImageSet(imset)
    
    print(f"[*] Stereomatching...")
    disparity = stereoMatch(imageSet)
    print(f"[*] Converting to pointcloud...")
    vertices,match = calculatePointCloud(disparity)
    
    print(f"[*] Removing errant points...")
    colorfilter = removeBackground(imageSet[0]).flatten()
    mask = np.logical_and(match,colorfilter)
    #mask = match
    
    color = np.reshape(cv.cvtColor(imageSet[0],cv.COLOR_BGR2RGB),(1920*1200,3))[mask,:]/255.
    viewer0 = plotPointCloud(vertices[:,mask],color,[0.,0.,-2.])
    zband = zBand(vertices,interactiveZBand(viewer0,vertices[:,mask]))
    viewer0.close()
    fitmask = np.logical_and(mask,zband)
    

    print(f"[*] Converting to cylindrical coordinates...")
    ccoord = cylindricalCoordinates(vertices)
    print(f"[*] Fitting surface model with RANSAC...")
    model,_ = ransacSinoidFit(ccoord[:,fitmask],VERBOSE=True)
    flattened = flattenSurface(ccoord,model)
    deviation = np.abs(flattened[2,mask].clip(-0.01,+0.01))

    viewer = plotPointCloud(vertices[:,mask],(color,deviation),[0.,0.,-2.])
    viewer2 = plotPointCloud(flattened[:,mask],(deviation,color),[0.,1.5,0.])

    print(f"Done!")



if __name__=="__main__":
    imageSets = listImageSets()
    imset = random.choice(imageSets)
    print(imset)
    demo(imset)
