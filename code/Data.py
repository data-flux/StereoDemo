#!/usr/bin/env python3
#
# Suggested use:
#   from Data import loadImageSet,listImageSets
#
from glob import glob
import cv2 as cv
import numpy as np
from parse import parse
from funcy import walk,lsplit,re_tester,rpartial


def loadImageSet(imageset,dir="./img/"):
    im1 = cv.imread(dir+imageset+"_L.jpg",1)
    im2 = cv.imread(dir+imageset+"_R.jpg",1)
    return im1,im2

    

def listImageSets(dir="./img/"):
    sets = sorted(glob(dir+"*_L.jpg"))
    sets = walk(lambda d:parse(dir+"{}_L.jpg",d)[0],sets)
    return sets


if __name__=="__main__":
    sets = listImageSets()
    print(sets)
    imageSet = loadImageSet(sets[26])
    cv.imshow('im1',imageSet[0])
    cv.imshow('im2',imageSet[1])
    while cv.waitKey(0)!=ord('q'):
        pass
