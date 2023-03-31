import warnings
import torchio as tio
import pandas as pd
import numpy as np
import napari
from pathlib import Path
import os
from tqdm import tqdm
import gc
import argparse
from deep_radiologist.utils import calculate_av_label_distance, head_print, subhead_print
from deep_radiologist.image_morph import resample_by_ratio, crop_3d_coords, crop_using_bbox, update_coords_after_crop
from deep_radiologist.data_loading import _load_point_data


def main(args):
    info = pd.read_csv(args.data_source_specifier_path)

    data_info_name = args.data_source_specifier_path.split('/')[-1].split('.')[0]

    n_rows = info.shape[0]

    if args.patch_size <= 0:
        steps_to_do = [True]
    else:
        print('args.patch_size was <= 0, so only generating whole patches')
        steps_to_do = [False, True]

    for whole in steps_to_do:
        # create training and testing data with, on average, x, y and z voxels between labels
        for v in args.voxel_spacings:
            # alongside cropped images, also always generate resampled whole volumes
            # for inference
            if whole:
                crop_buffer = None
                patch_size = None
                folder_prefix = 'whole'
            else:
                crop_buffer = 16
                patch_size = args.patch_size
                folder_prefix = 'cropped'

            for i in range(n_rows):
                img = info.loc[i, 'image_file_path']
                label = info.loc[i, 'label_file_path']
                split = info.loc[i, 'split']

                out_label_dir = f'./dataset/{data_info_name}/{folder_prefix}/{split}_labels_{str(v)}/'
                out_image_dir = f'./dataset/{data_info_name}/{folder_prefix}/{split}_images_{str(v)}/'

                # make directories if they don't exist
                Path(out_label_dir).mkdir(parents=True, exist_ok=True)
                Path(out_image_dir).mkdir(parents=True, exist_ok=True)

                p = Path(img)
                filename = p.stem
                out_path = out_label_dir + filename
                image_out_path = out_image_dir + filename

                if os.path.isdir(img):
                    # swap x and y axes on dicom images
                    swap_xy = True
                else:
                    swap_xy = False

                if (patch_size is None and not os.path.isfile(f'{image_out_path}-image.nii')) \
                    or (patch_size is not None and not os.path.isfile(f'{image_out_path}-0-image.nii')):

                    head_print('Starting conversion of ' + filename)

                    img = tio.ScalarImage(img, check_nans=True)

                    subhead_print('Resampling the image')

                    label_locations = _load_point_data(
                        label,
                        swap_xy,
                        file_type='mctv_mat',
                        label_name=args.label_name
                    )
                    if args.file_type == 'mctv_mat':
                        other_label_locations = 

                    TODO:
                        - make a matlab mctv to csv file converter function
                        - make this function read in csv files instead of matlab files
                        - make two inputs: path to file with labels, path to file/s with labels of checked area
                        - add a check to raise an error if cropped patch size is too close to whole image size and provide a recommendation for the path they should give to the config file.
                        - simplify this function

                    resample_ratio = calculate_av_label_distance(label_locations) / v

                    subhead_print(f'Resampling image by ratio: {resample_ratio}')
                    img = resample_by_ratio(img, resample_ratio)
                    subhead_print(f'Resampling labels by ratio: {resample_ratio}')
                    label_locations = np.rint(label_locations / resample_ratio)

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
                        cornea_locations = cornea_locations.T
                        rhabdom_locations = rhabdom_locations.T

                    subhead_print('Saving')
                    # convert img to uint16 to save on space
                    subject.img.set_data(subject.img.data.numpy().astype(np.uint16))

                    if patch_size is None:
                        im_path = image_out_path + '-image.nii'
                        if not os.path.isfile(im_path):
                            print('saving image to ' + im_path)
                            subject.img.save(im_path)
                        if 'corneas' in args.label_name:
                            np.savetxt(out_path + '-corneas.csv', cornea_locations, delimiter=",")
                        if 'rhabdoms' in args.label_name:
                            np.savetxt(out_path + '-rhabdoms.csv', rhabdom_locations, delimiter=",")
                        with open(out_path + '-resample_ratio.txt', 'w') as f:
                            f.write(str(resample_ratio))
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
                                image_path = f'{image_out_path}-{i}-image.nii'
                                cornea_path = f'{out_path}-{i}-corneas.csv'
                                rhabdom_path = f'{out_path}-{i}-rhabdoms.csv'

                                if not os.path.isfile(image_path):
                                    patch.img.save(image_path)
                                if not os.path.isfile(cornea_path) and 'corneas' in args.label_name:
                                    corneas_in_patch = crop_3d_coords(cornea_locations, patch_locations[i]) 
                                    np.savetxt(cornea_path, corneas_in_patch, delimiter=",")
                                    # debug plot
                                    # viewr = napari.view_image(patch.img.numpy(), name='image patch', ndisplay=3)
                                    # viewr.add_points(corneas_in_patch, name='corneas in patch')
                                    # viewr.add_points(cornea_locations, name='all corneas')
                                if not os.path.isfile(rhabdom_path) and 'rhabdoms' in args.label_name:
                                    rhabdoms_in_patch = crop_3d_coords(rhabdom_locations, patch_locations[i]) 
                                    np.savetxt(rhabdom_path, rhabdoms_in_patch, delimiter=",")
                        except:
                            warnings.warn(
                                f'Patch size of {patch_size} is larger than image size of {subject.img.shape}'
                                ' so image patch was not generated. '
                                'Consider reducing patch size, using a larger resolution image or don\'t use patches at all (just the whole images)'
                            )

                    # run the garbage collector to make sure memory is freed
                    gc.collect()
                else:
                    print(out_path + ' has already been created, so skipping')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate nicely resampled and formatted images and labels from the datas raw format.'
    )

    parser.add_argument(
        'data_source_specifier_path',
        type=str,
        help='''
        The path to your .csv file that specifies where your images and labels are and whether it should be
        used for training or testing.
        This .csv file should have the following columns:
        1. image_file_path (path to the dicom or nifti file used to annotate with mctv)
        2. label_file_path (matlab files with corneas and rhabdoms generated by mctv)
        3. split (containing 'train' or 'test' to say how the data should be split)
            This should be generated randomly, or rerun multiple times as part of a
            k-fold cross validation process.

        Example:
            data_source_specifiers/fiddler.csv
            or 
            data_source_specifiers/paraphronima.csv
        '''
    )

    parser.add_argument(
        '--voxel_spacings',
        '-v',
        nargs='+',
        type=int,
        help='''
        The voxel spacings you would like between your labels to use to resample your dataset.
        This is resample size that gives you the average spacing between labels.
        Can be a single value or multiple values.

        One thing to consider in this decision is that an image with too few voxels may not provide enough information for the model to detect the
        feature, whereas a image with too many voxels may have so much information that it cannot be loaded into your computer's memory, or require an unreasonably large training time for your model. If you face memory usage issues during training or inference, consider reducing the voxel spacing used.

        Example:
            -v 10
            or
            -v 10 20 25
        ''',
        default=[10, 20]
    )

    parser.add_argument(
        '--label_name',
        '-l',
        nargs='+',
        type=str,
        help='''
        The name of the label or labels you would like to use.

        Example:
            -l corneas
            or
            -l rhabdoms
            or
            -l corneas rhabdoms

        ''',
        default=['corneas', 'rhabdoms']
    )

    parser.add_argument(
        '--patch_size',
        '-p',
        type=int,
        help='''
        The patch size to crop your images into smaller sizes that can be loaded efficiently.
        If <= 0 is supplied then only whole images and labels will be generated.
        Cropped images and labels will be placed at 'dataset/your_config_name/cropped/',
        while whole images (useful for inference and testing) will be placed at
        'dataset/your_config_name/whole/'.

        Example:
            -p 0
            or
            -p 256
        ''',
        default=256
    )

    args = parser.parse_args()

    if 'corneas' not in args.label_name and 'rhabdoms' not in args.label_name:
        raise NotImplementedError(
            'Unknown label name detected. Currently, only corneas and rhabdoms '
            'from mctv .mat files are supported. '
            'If you would like support for another file type, please open an issue at '
            'https://www.github.com/jakemanger/deepradiologist'
        )

    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
