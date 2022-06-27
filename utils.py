# -*- coding: utf-8 -*-
"""
@author: Trinh Pham <phkhanhtrinh23@gmail.com>
"""

import numpy as np
import cv2


def get_pixels_hu(scans):
    # image.shape: (129,512,512)
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def transform_ctdata(image, windowWidth, windowCenter, normal=False):
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (image - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


def getRangImageDepth(image):
    firstflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[0]):
        notzeroflag = np.max(image[z])
        if notzeroflag and firstflag:
            startposition = z
            firstflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition


def clahe_equalized(imgs,start,end):
   assert (len(imgs.shape)==3)
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   imgs_equalized = np.empty(imgs.shape)
   for i in range(start, end+1):
       imgs_equalized[i,:,:] = clahe.apply(np.array(imgs[i,:,:], dtype = np.uint8))
   return imgs_equalized