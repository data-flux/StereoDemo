#!/usr/bin/env python3
#
# Suggested use:
#   from Filter import removeBackground zBand
#
import cv2 as cv
import numpy as np

def removeBackground(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    shape = img.shape[:2]
    mask = np.zeros((shape[0]+2,shape[1]+2),dtype=np.uint8)
    def floodfill(seed,mask):
        _, _, mask, _ = cv.floodFill(
                img,                                # input image
                mask,                               # input mask, zeros
                seed,                               # starting point
                None,                               # fill value, not used
                [10, 10, 50],                       # how much darker is allowed
                [10, 80, 50],                       # how much lighter is allowed
                8|cv.FLOODFILL_MASK_ONLY|(255<<8)   # 8-connectivity, mask only, fill value 255
            )
        return mask
    mask = floodfill((shape[1]//2,shape[0]//2),mask)
    while np.sum(mask)<(1_000*255):
        rand = np.random.randint(-5,5,size=2)
        mask = floodfill((shape[1]//2+rand[0],shape[0]//2+rand[1]),mask)
    #perform morphological closing
    mask = 255-mask[1:-1,1:-1]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(21,21))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)
    return mask>0

def zBand(pointcloud,z_range):
    import warnings
    with warnings.catch_warnings():
        # suppress a warning over possible NaN still in ccoord
        warnings.simplefilter("ignore")
        zband = np.logical_and(min(z_range)<pointcloud[2,:], pointcloud[2,:]<max(z_range))
    return zband

def interactiveZBand(viewer,vertices):
    print("        Select Z-range for model fitting with <Ctrl>+<LMB>\n        Cancel selection with <Ctrl>+<RMB>\n        Confirm selection with <Return>")
    viewer.wait()
    sel = viewer.get('selected')
    return (min(vertices[2,sel]),max(vertices[2,sel]))


if __name__=="__main__":
    from Data import listImageSets, loadImageSet
    img = loadImageSet(listImageSets()[34])[0]
    #img = cv.resize(img,(960,600))
    cv.imshow("orig",img)
    image = removeBackground(img)
    cv.imshow("filled",image.astype(np.float32))
    while cv.waitKey(0) != ord('q'):
        pass
