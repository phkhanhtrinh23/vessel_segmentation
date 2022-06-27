# -*- coding: utf-8 -*-
"""
@author: Trinh Pham <phkhanhtrinh23@gmail.com>
"""

import numpy as np
import pydicom
import os
from keras.preprocessing.image import ImageDataGenerator
from utils import *
from generator import *
import cv2

seed=1
data_gen_args = dict(rotation_range=3,
                    width_shift_range=0.01,
                    height_shift_range=0.01,
                    shear_range=0.01,
                    zoom_range=0.01,
                    fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
print('ImageDataGenerator has finished.')

output_path = "./data/artery/train_artery.h5"

if os.path.exists(output_path):
  os.remove(output_path)
dataset = DatasetWriter(image_dims=(1660, 512, 512, 1),
                        mask_dims=(1660, 512, 512, 1),
                        output_path=output_path)

print('DatasetWriter has finished.')

for i in range(1, 12):
  list_images = []
  list_vein = []

  data_path = f'./data/3Dircadb1/3Dircadb1.{i}/PATIENT_DICOM/'
  label_path = f'./data/3Dircadb1/3Dircadb1.{i}/MASKS_DICOM/artery'

  if not os.path.isdir(label_path):
    continue

  image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path)]
  image_slices.sort(key=lambda x: int(x.InstanceNumber))

  vein_slices = [pydicom.dcmread(label_path + '/' + s) for s in os.listdir(label_path)]
  vein_slices.sort(key=lambda x: int(x.InstanceNumber))
  vein = np.stack([s.pixel_array for s in vein_slices])
  
  images = get_pixels_hu(image_slices)
  images = transform_ctdata(images, 500, 150)
  start, end = getRangImageDepth(vein)
  images = clahe_equalized(images, start, end)
  images /= 255.
  
  # Extract only the slices that contain 
  # the artery among all the slices, the rest do not
  total = (end - 4) - (start + 4) + 1
  print(f"The {i}-th person, total slices {total}")

  images = images[start + 5:end - 5]
  print(f"The {i}-th person, images.shape: {images.shape}")

  vein[vein > 0] = 1

  vein = vein[(start + 5) : (end - 5)]

  list_images.append(images)
  list_vein.append(vein)

  list_images = np.vstack(list_images)
  list_images = np.expand_dims(list_images, axis=-1)

  list_vein = np.vstack(list_vein)
  list_vein = np.expand_dims(list_vein, axis=-1)

  image_datagen.fit(list_images, augment=True, seed=seed)
  mask_datagen.fit(list_vein, augment=True, seed=seed)

  image_generator = image_datagen.flow(list_images, seed=seed)
  mask_generator = mask_datagen.flow(list_vein, seed=seed)

  train_generator = zip(image_generator, mask_generator)
  x = []
  y = []
  i = 0
  for x_batch, y_batch in train_generator:
      i += 1
      x.append(x_batch)
      y.append(y_batch)
      if i >= 2:
          break
  x = np.vstack(x)
  y = np.vstack(y)

  dataset.add(list_images, list_vein)
  dataset.add(x, y)
  print('Adding has finished.\n')

print(f"The number of the total images in train_data are {dataset.close()}")