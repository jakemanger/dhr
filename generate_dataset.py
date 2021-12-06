"""
Creates the dataset images (the volumes) and labels (of corneas and rhabdoms)
at 4 different resolutions (average of 10, 15, 20 and 25 voxels between corneas).
Uses image and annotation file paths found in the data_info.csv file to generate dataset.
Note, it may also be a good idea to 
"""

import warnings
from sklearn.utils import resample
import torchio as tio
import pandas as pd
import numpy as np
import napari
from pathlib import Path
import os

from mctnet.label_generation import create_annotated_volumes
from mctnet.utils import calculate_av_cornea_distance, head_print, subhead_print
from mctnet.image_morph import resample_by_ratio

patch_size = 256

if __name__ == '__main__':
    info = pd.read_csv('data_info.csv')

    n_rows = info.shape[0]

    for v in [10, 15, 20, 25]:
        out_label_dir = f'./dataset/labels_{str(v)}/'
        out_image_dir = f'./dataset/images_{str(v)}/'

        for i in range(n_rows):
            img = info.loc[i, 'image_file_path']
            label = info.loc[i, 'label_file_path']

            p = Path(img)
            filename = p.stem
            out_path = out_label_dir + filename
            image_out_path = out_image_dir + filename

            if os.path.isdir(img):
                swap_xy = True
            else:
                swap_xy = False
            
            if not os.path.isfile(f'{out_path}-corneas.nii') \
                and not os.path.isfile(f'{out_path}-rhabdoms.nii') \
                    and not os.path.isfile(f'{image_out_path}-image.nii') \
                        and not os.path.isfile(f'{image_out_path}-0-image.nii.gz'):

                head_print('Starting conversion of ' + filename)
                            
                img = tio.ScalarImage(img, check_nans=True)
                
                subhead_print('Resampling the image')

                print('Calculating average distance between corneas to resample accordingly')
                resample_ratio = calculate_av_cornea_distance(label) / v

                if abs(resample_ratio) > 0.5 and abs(resample_ratio) < 1.5:
                    img = resample_by_ratio(img, resample_ratio)
                    
                    subhead_print('Creating annotated volumes')
                    corneas, rhabdoms = create_annotated_volumes(
                        label,
                        img.data.numpy(),
                        swap_xy,
                        resample_ratio=resample_ratio
                    )

                    corneas = tio.Image(
                        tensor=corneas,
                        affine=img.affine,
                        orientation=img.orientation,
                        spacing=img.spacing,
                        type=tio.LABEL,
                        check_nans=True
                    )
                    rhabdoms = tio.Image(
                        tensor=rhabdoms,
                        affine=img.affine,
                        orientation=img.orientation,
                        spacing=img.spacing,
                        type=tio.LABEL,
                        check_nans=True
                    )
                    
                    assert corneas.shape == img.shape, 'Cornea annotation and image shape mismatch'
                    assert rhabdoms.shape == img.shape, 'Rhabdom annotation and image shape mismatch'

                    subhead_print('Saving')
                    # convert img to uint16 to save on space
                    img.set_data(img.data.numpy().astype(np.uint16))
                    # convert annotations to float32 to save on space
                    corneas.set_data(corneas.data.numpy().astype(np.float32))
                    rhabdoms.set_data(rhabdoms.data.numpy().astype(np.float32))

                    if patch_size is None:
                        im_path = image_out_path + '-image.nii'
                        if not os.path.isfile(im_path):
                            print('saving image to ' + im_path)
                            img.save(im_path)
                        corneas.save(out_path + '-corneas.nii')
                        rhabdoms.save(out_path + '-rhabdoms.nii')
                    else:
                        # create subject
                        subject = tio.Subject(
                            image=img,
                            corneas=corneas,
                            rhabdoms=rhabdoms
                        )

                        # now divide into patches, so data is easy to read
                        try:
                            sampler = tio.GridSampler(subject=subject, patch_size=patch_size)
                            num_patches = len(sampler)
                            # and save each patch
                            for i, patch in enumerate(sampler):
                                image_path = f'{image_out_path}-{i}-image.nii.gz'
                                cornea_path = f'{out_path}-{i}-corneas.nii.gz'
                                rhabdom_path = f'{out_path}-{i}-rhabdoms.nii.gz'
                                if not os.path.isfile(image_path):
                                    patch.image.save(image_path)
                                if not os.path.isfile(cornea_path):
                                    patch.corneas.save(cornea_path)
                                if not os.path.isfile(rhabdom_path):
                                    patch.rhabdoms.save(rhabdom_path)
                                print(f'Saved patch {i} or {num_patches}')
                        except:
                            warnings.warn(f'Patch size of {patch_size} is larger than image size of {img.size}')
                else:
                    warnings.warn(
                        f'Resample ratio is greater than +/-50%. A different resampled resolution is likely' \
                        ' more suitable for this scan. Skipping.'
                    )
            else:
                print(out_path + ' has already been created, so skipping')