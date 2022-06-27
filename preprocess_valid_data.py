# -*- coding: utf-8 -*-
"""
@author: Trinh Pham <phkhanhtrinh23@gmail.com>
"""

import numpy as np
import pydicom
import os
from generator import *
from utils import *

output_path = "./data/artery/valid_artery.h5"

if os.path.exists(output_path):
  os.remove(output_path)

list_images = []
list_vein = []

for i in range(12, 21):
  data_path = f'./data/3Dircadb1/3Dircadb1.{i}/PATIENT_DICOM/'
  label_path = f'./data/3Dircadb1/3Dircadb1.{i}/MASKS_DICOM/artery'

  if not os.path.isdir(label_path):
    continue

  vein_slices = [pydicom.dcmread(label_path + '/' + s) for s in os.listdir(label_path)]
  vein_slices.sort(key = lambda x: int(x.InstanceNumber))

  vein = np.stack([s.pixel_array for s in vein_slices])
  
  image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path)]
  image_slices.sort(key = lambda x: int(x.InstanceNumber))
  
  images = get_pixels_hu(image_slices)
  images = transform_ctdata(images,500,150)
  start,end = getRangImageDepth(vein)
  images = clahe_equalized(images,start,end)
  images /= 255.

  total = (end - 4) - (start+4) +1
  print(f"The {i}-th person, total slices {total}")

  images = images[start+5:end-5]
  print(f"The {i}-th person, images.shape: {images.shape}")

  vein[vein > 0] = 1
  vein = vein[(start+5) : (end-5)]

  list_images.append(images)
  list_vein.append(vein)
    
list_images = np.vstack(list_images)
list_images = np.expand_dims(list_images,axis=-1)

list_vein = np.vstack(list_vein)
list_vein = np.expand_dims(list_vein,axis=-1)

dataset = DatasetWriter(image_dims=(list_images.shape[0], list_images.shape[1], list_images.shape[2], 1),
                        mask_dims=(list_images.shape[0], list_images.shape[1], list_images.shape[2], 1),
                        output_path=output_path)

dataset.add(list_images, list_vein)

print('Adding has finished.')

print(f"The number of the total images in valid_data are {dataset.close()}")