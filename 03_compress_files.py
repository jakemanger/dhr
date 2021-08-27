# a script that loads up all the images and labels and converts them to .nii.gz files

import os
from pathlib import Path
import numpy as np
import nibabel as nib

img_dir = './dataset/images/'
label_dir = './dataset/labels'

imgs = []
labels = []

for file in os.listdir(img_dir):
    if file.endswith('.nii'):
        imgs.append(img_dir + file)
        
for file in os.listdir(label_dir):
    if file.endswith('.nii'):
        labels.append(label_dir + file)

print(f'Converting {len(imgs)} images and {len(labels)} labels')

to_convert = imgs + labels

# print(to_convert)

for i in to_convert:
    print(f'Starting conversion of {i}')
    image = nib.load(i)
    nib.save(image, f'{i}.gz')
    print(f'Success! {i}.gz is converted')