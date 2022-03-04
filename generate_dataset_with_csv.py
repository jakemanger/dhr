"""
Creates the dataset images (the volumes) and labels (of corneas and rhabdoms)
at 4 different resolutions (average of 10, 15, 20 and 25 voxels between corneas).
Uses image and annotation file paths found in the data_info.csv file to generate dataset.
Note, it may also be a good idea to 
"""

import warnings
import torchio as tio
import pandas as pd
import numpy as np
import napari
from pathlib import Path
import os
from tqdm import tqdm
import gc

from mctnet.label_generation import create_annotated_volumes
from mctnet.utils import calculate_av_cornea_distance, head_print, subhead_print
from mctnet.image_morph import resample_by_ratio, crop_3d_coords, crop_using_bbox, update_coords_after_crop
from mctnet.data_loading import _load_point_data

# patches
patch_size = 256
# whole images
# patch_size = None

# crop images to edges of labels with a given buffer
crop_buffer = 16
# no crop
# crop_buffer = None

if __name__ == '__main__':
    info = pd.read_csv('data_info.csv')

    n_rows = info.shape[0]

    # for v in [10, 15, 20, 25]:
    for v in [10, 20, 25]:
        if crop_buffer is not None:
            out_label_dir = f'./dataset/all/cropped_with_csv_labs/labels_{str(v)}/'
            out_image_dir = f'./dataset/all/cropped_with_csv_labs/images_{str(v)}/'
        else:
            out_label_dir = f'./dataset/all/labels_{str(v)}/'
            out_image_dir = f'./dataset/all/images_{str(v)}/'

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
            
            if not os.path.isfile(f'{image_out_path}-image.nii') \
                and not os.path.isfile(f'{image_out_path}-0-image.nii.gz'):

                head_print('Starting conversion of ' + filename)
                            
                img = tio.ScalarImage(img, check_nans=True)
                
                subhead_print('Resampling the image')

                print('Calculating average distance between corneas to resample accordingly')
                resample_ratio = calculate_av_cornea_distance(label) / v
                print(f'Resampling volume by ratio: {resample_ratio}')

                if abs(resample_ratio) > 0.5 and abs(resample_ratio) < 2:
                    img = resample_by_ratio(img, resample_ratio)

                    subhead_print('Creating annotated volumes')
                    cornea_locations, rhabdom_locations = _load_point_data(label, swap_xy)
                    # apply the same resampling that was made to the image, so that 
                    # annotated features line up correctly
                    cornea_locations = np.rint(cornea_locations / resample_ratio)
                    cornea_locations = np.array([
                        cornea_locations[:, 2],
                        cornea_locations[:, 1],
                        cornea_locations[:, 0]
                    ], dtype=np.int)

                    rhabdom_locations = np.rint(rhabdom_locations / resample_ratio)
                    rhabdom_locations = np.array([
                        rhabdom_locations[:, 2],
                        rhabdom_locations[:, 1],
                        rhabdom_locations[:, 0]
                    ], dtype=np.int)

                    # now do crop to edges of corneas and rhabdoms
                    if crop_buffer is not None:
                        subhead_print('Cropping to edges of corneas and rhabdoms with buffer')

                        subject = tio.Subject(img=img)

                        # viewr = napari.view_image(subject.img.numpy(), name='image', ndisplay=3)

                        # find edges of corneas and rhabdoms, incorporate buffer and make mask
                        all_labs = np.concatenate([cornea_locations, rhabdom_locations], axis=1)

                        min_vals = (0, 0, 0)
                        max_vals = subject.img.shape[1:]

                        bbox = (
                            np.maximum(np.min(all_labs, axis=1) - crop_buffer, min_vals),
                            np.minimum(np.max(all_labs, axis=1) + crop_buffer + 1, max_vals)
                        )

                        subject = crop_using_bbox(
                            subject,
                            bbox,
                        )
                        
                        cornea_locations = update_coords_after_crop(cornea_locations, bbox)
                        rhabdom_locations = update_coords_after_crop(rhabdom_locations, bbox)
                        cornea_locations = cornea_locations.T
                        rhabdom_locations = rhabdom_locations.T

                        del(all_labs)

                        # viewr.add_image(subject.img.numpy(), name='cropped image')
                        # viewr.add_points(cornea_locations, name='corneas')
                        # viewr.add_points(rhabdom_locations, name='rhabdoms')
                    else:
                        # create subject
                        subject = tio.Subject(img=img)

                    subhead_print('Saving')
                    # convert img to uint16 to save on space
                    subject.img.set_data(subject.img.data.numpy().astype(np.uint16))

                    if patch_size is None:
                        im_path = image_out_path + '-image.nii'
                        if not os.path.isfile(im_path):
                            print('saving image to ' + im_path)
                            subject.img.save(im_path + '.gz')
                        np.savetxt(out_path + '-corneas.csv', cornea_locations, delimiter=",")
                        np.savetxt(out_path + '-rhabdoms.csv', rhabdom_locations, delimiter=",")
                    else:

                        # now divide into patches, so data is easy to read
                        try:
                            sampler = tio.GridSampler(subject=subject, patch_size=patch_size)
                            num_patches = len(sampler)

                            # and save each patch
                            print(f"saving {str(num_patches)} patches...")
                            
                            patch_locations = sampler.locations

                            # save patch images
                            for i, patch in tqdm(enumerate(sampler), total=num_patches):
                                image_path = f'{image_out_path}-{i}-image.nii.gz'
                                cornea_path = f'{out_path}-{i}-corneas.csv'
                                rhabdom_path = f'{out_path}-{i}-rhabdoms.csv'

                                if not os.path.isfile(image_path):
                                    patch.img.save(image_path)
                                if not os.path.isfile(cornea_path):
                                    corneas_in_patch = crop_3d_coords(cornea_locations, patch_locations[i]) 
                                    np.savetxt(cornea_path, corneas_in_patch, delimiter=",")
                                    # debug plot
                                    # viewr = napari.view_image(patch.img.numpy(), name='image patch', ndisplay=3)
                                    # viewr.add_points(corneas_in_patch, name='corneas in patch')
                                    # viewr.add_points(cornea_locations, name='all corneas')
                                if not os.path.isfile(rhabdom_path):
                                    rhabdoms_in_patch = crop_3d_coords(rhabdom_locations, patch_locations[i]) 
                                    np.savetxt(rhabdom_path, rhabdoms_in_patch, delimiter=",")
                        except:
                            warnings.warn(f'Patch size of {patch_size} is larger than image size of {subject.img.shape}')

                    # run the garbage collector to make sure memory is freed
                    gc.collect()
                else:
                    warnings.warn(
                        f'Resample ratio is greater than +/-50%. A different resampled resolution is likely' \
                        ' more suitable for this scan. Skipping.'
                    )
            else:
                print(out_path + ' has already been created, so skipping')