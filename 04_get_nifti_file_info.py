import os
import nibabel as nib

# image_dir = './dataset/images/'

# imgs = []

# for file in os.listdir(image_dir):
#     if file.endswith('.nii'):
#         imgs.append(image_dir + file)

imgs = ['/Volumes/1tb_ssd/mctv_analysis/Head Scans/Platyscelus_02_FEG191112_082.nii']

for img in imgs:
    proxy_img = nib.load(img)
    print(f'\nImage: {img}')
    print(f'Shape: {proxy_img.shape}')
    print(f'Header: {proxy_img.header.get_zooms()}')