import os
from pydicom import dcmread
from pydicom.filereader import read_file_meta_info, read_dicomdir
import os
import nibabel as nib
import torchio as tio

info = pd.read_csv('missing_scan_spacing.csv')

scan_names = info.scan_name

niftis = []
dicoms = []

for name in scan_names:
    if name.endswith('.nii'):
        niftis.append(name)
    else:
        dicoms.append(name)
    
for img in dicoms:
    proxy_img = dcmread(img)
    
for img in niftis:
    proxy_img = nib.load(img)
