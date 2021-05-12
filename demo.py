#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import random
import os
from matplotlib import pyplot as pp
from funcy import lmap
from code.StereoMatch import stereoMatch, calculatePointCloud, plotPointCloud
from code.Filter import removeBackground, zBand, interactiveZBand
from code.Data import loadImageSet, listImageSets
from code.Surface import cylindricalCoordinates, ransacSinoidFit, flattenSurface
from code.Anomaly import highlightAnomalies


try:
    import pptk
    INTERACTIVE = True
except:
    print("Could not import module pptk, running non-interactively!")
    INTERACTIVE = False

def demo(imset):
    print(f"[1/5] Image Acquisition")
    imageSet = loadImageSet(imset)

    try:
        os.mkdir(f"output/{imset}")
    except FileExistsError:
        pass
    cv.imwrite(f"output/{imset}/00_reference.jpg",imageSet[0])
    print(f"        Wrote reference image to output/{imset}/00_reference.jpg")
    
    print(f"[2/5] Semi-Global Stereo Matching")
    disparity = stereoMatch(imageSet)
    disparity_im = np.zeros([1200,1920,3],dtype=np.float32)
    disparity_im[~np.isnan(disparity),0] = disparity[~np.isnan(disparity)]
    disparity_im[~np.isnan(disparity),1] = disparity[~np.isnan(disparity)]
    disparity_im[~np.isnan(disparity),2] = disparity[~np.isnan(disparity)]
    disparity_im[np.isnan(disparity),:] = [220,0,220]
    cv.imwrite(f"output/{imset}/01_disparity.jpg",disparity_im)
    print(f"        Wrote disparity image to output/{imset}/01_disparity.jpg")

    print(f"[3/5] 3D Geometry Reconstruction")
    vertices,match = calculatePointCloud(disparity)
    
    print(f"[4/5] Robust Pipe Surface Fitting")
    print(f"      Removing errant points")
    colorfilter = removeBackground(imageSet[0]).flatten()
    mask = np.logical_and(match,colorfilter)
    
    color = np.reshape(cv.cvtColor(imageSet[0],cv.COLOR_BGR2RGB),(1920*1200,3))[mask,:]/255.
    color8b = np.reshape(cv.cvtColor(imageSet[0],cv.COLOR_BGR2RGB),(1920*1200,3))[mask,:]
    if INTERACTIVE:
        viewer0 = plotPointCloud(vertices[:,mask],color,[0.,0.,-2.],theta=0)
        while True:
            try:
                zband = zBand(vertices,interactiveZBand(viewer0,vertices[:,mask]))
                break
            except ValueError:
                print("Invalid selection, please select at least two points for a valid Z-range")
            except KeyboardInterrupt:
                raise KeyboardInterrupt
        viewer0.close()
    else:
        zband = zBand(vertices,(-1.5,-2.0))
    fitmask = np.logical_and(mask,zband)
    

    print(f"      Conversion to cylindrical coordinates")
    ccoord = cylindricalCoordinates(vertices)
    print(f"      RANSAC Fourier series approximation")
    model,_ = ransacSinoidFit(ccoord[:,fitmask],VERBOSE=True)
    flattened = flattenSurface(ccoord,model)

    print(f"[5/5] Anomaly Detection and Processing") 
    deviation = np.abs(flattened[2,mask].clip(-0.02,+0.02))
    if INTERACTIVE:
        viewer1 = plotPointCloud(vertices[:,mask],(deviation,color),[0.,0.,-2.])
    pc = np.hstack([vertices[:,mask].T,flattened[[[2]],mask].T,color8b])
    np.save(f"output/{imset}/02_pointcloud.npy",pc)
    print(f"        Wrote pointcloud to output/{imset}/02_pointcloud.npy")
    if INTERACTIVE:
        viewer2 = plotPointCloud(flattened[:,mask],(deviation,color),[0.,1.5,0.])
    pc2 = np.hstack([flattened[[[2]],mask].T,flattened[:2,mask].T,color8b])
    np.save(f"output/{imset}/03_fit.npy",pc2)
    print(f"        Wrote fit pointcloud to output/{imset}/03_fit.npy")

    highlighted = highlightAnomalies(imageSet[0],flattened[2,:],mask)
    cv.imwrite(f"output/{imset}/04_highlighted.jpg",highlighted)
    print(f"        Wrote highlighted image to output/{imset}/04_highlighted.jpg")
    

    fit = pc2
    sel = np.logical_and(np.logical_and(1.00<fit[:,2],fit[:,2]<2.5),np.abs(fit[:,0])<0.05)
    limits = [np.min(fit[sel,1]),
              np.max(fit[sel,1]),
              np.min(fit[sel,2]),
              np.max(fit[sel,2])]
    
    pp.scatter(-fit[sel,1],fit[sel,2],s=1,c=fit[sel,3:6]/255.,marker='.')
    pp.xlim(limits[0],limits[1])
    pp.ylim(limits[2],limits[3])
    pp.axis("off")
    pp.tight_layout()
    pp.savefig(f"output/{imset}/05_unfolded.jpg",dpi=300,pad_inches=0)
    print(f"        Wrote unfolded image to output/{imset}/05_unfolded.jpg")
    pp.scatter(-fit[sel,1],fit[sel,2],s=1,c=fit[sel,0],marker='.',cmap='seismic',vmin=-0.025,vmax=0.025)
    pp.xlim(limits[0],limits[1])
    pp.ylim(limits[2],limits[3])
    pp.axis("off")
    pp.tight_layout()
    pp.savefig(f"output/{imset}/06_anomaly.jpg",dpi=300,pad_inches=0)
    print(f"        Wrote anomaly image to output/{imset}/06_anomaly.jpg")



    print(f"Done!")




if __name__=="__main__":
    imageSets = listImageSets()
    print(f"Found {len(imageSets)} image sets")
    while True:
        print(f"(L)ist, (R)andom, or enter a number 1-{len(imageSets)} to choose a set.")
        inp = input()
        if inp=="L" or inp=="l":
            for i,im in enumerate(imageSets):
                print(f"{i+1:2d}: {im}")
            continue
        elif inp=="R" or inp=='r':
            imset = random.choice(imageSets)
            break
        try:
            choice = int(inp)
            imset = imageSets[choice-1]
            break
        except:
            pass
    print(f"Running pipeline on image set {imset}...")
    demo(imset)
