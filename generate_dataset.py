"""
Creates the dataset images (the volumes) and labels (of corneas and rhabdoms)
at 4 different resolutions (average of 10, 15, 20 and 25 voxels between corneas).
Uses image and annotation file paths found in the data_info.csv file to generate dataset.
Note, it may also be a good idea to 
"""

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
            image_out_path = out_image_dir + filename + '.nii'

            if os.path.isdir(img):
                swap_xy = True
            else:
                swap_xy = False
            
            if not os.path.isfile(f'{out_path}_corneas.nii') \
                or not os.path.isfile(f'{out_path}_rhabdoms.nii') \
                    or not os.path.isfile(image_out_path):
                head_print('Starting conversion of ' + filename)
                            
                img = tio.ScalarImage(img)
                
                subhead_print('Resampling the image')

                print('Calculating average distance between corneas to resample accordingly')
                resample_ratio = calculate_av_cornea_distance(label) / v

                img = resample_by_ratio(img, resample_ratio)
                
                subhead_print('Creating annotated volumes')
                corneas, rhabdoms = create_annotated_volumes(
                    label,
                    img.data.numpy(),
                    swap_xy,
                    resample_ratio=resample_ratio
                )

                corneas = tio.LabelMap(
                    tensor=corneas,
                    affine=img.affine,
                    orientation=img.orientation,
                    spacing=img.spacing
                )
                rhabdoms = tio.LabelMap(
                    tensor=rhabdoms,
                    affine=img.affine,
                    orientation=img.orientation,
                    spacing=img.spacing
                )
                
                assert corneas.shape == img.shape, 'Cornea annotation and image shape mismatch'
                assert rhabdoms.shape == img.shape, 'Rhabdom annotation and image shape mismatch'

                subhead_print('Saving')
                # convert img to uint16 to save on space
                img.set_data(img.data.numpy().astype(np.uint16))

                if not os.path.isfile(image_out_path):
                    print('saving image to ' + image_out_path)
                    img.save(image_out_path)

                del(img)

                # convert annotations to float32 to save on space
                print('saving labels to ' + out_path)

                corneas.set_data(corneas.data.numpy().astype(np.float32))
                corneas.save(out_path + '_corneas.nii')

                del(corneas)

                rhabdoms.set_data(rhabdoms.data.numpy().astype(np.float32))
                rhabdoms.save(out_path + '_rhabdoms.nii')

                del(rhabdoms)
            else:
                print(out_path + ' has already been created, so skipping')