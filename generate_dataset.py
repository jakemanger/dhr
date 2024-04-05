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
from deep_radiologist.utils import (
    calculate_av_label_distance, head_print, subhead_print
)
from deep_radiologist.image_morph import (
    resample_by_ratio, crop_3d_coords, crop_using_bbox, update_coords_after_crop
)
from deep_radiologist.data_loading import _load_point_data


class DatasetGenerator:
    def __init__(self, args):
        ''' Initializes the dataset generator

        Args:
            args (argparse.Namespace): The parsed arguments
        '''

        self.args = args
        self.info = pd.read_csv(args.data_source_specifier_path)
        self.data_info_name = (
            args.data_source_specifier_path
            .split('/')[-1]
            .split('.')[0]
        )
        self.n_rows = self.info.shape[0]
        if self.args.patch_size <= 0:
            print('args.patch_size was <= 0, so only generating whole patches')
            self.whole_patches = [True]
            self.patch_size = None
        else:
            self.whole_patches = [False, True]
            self.patch_size = self.args.patch_size

    def run(self):
        ''' Runs the dataset generation process '''

        for whole in self.whole_patches:
            if whole:
                self.folder_prefix = 'whole'
            else:
                self.folder_prefix = 'patches'
            self._process_images(whole)

    def _process_images(self, whole):
        ''' Processes the images

        Args:
            whole (bool): Whether to process whole images or cropped images
        '''

        for v in self.args.voxel_spacings:
            if v is None:
                v = self._measure_average_voxel_spacing()

            print(f'Processing images with voxel spacing {v}')
            for i in tqdm(range(self.n_rows)):
                self._process_single_image(
                    i,
                    v,
                    whole=whole,
                    crop=True
                )

    def _measure_average_voxel_spacing(self):
        print('Calculating average voxel spacing from input data')

        resample_ratios = list()

        for i in tqdm(range(self.n_rows)):
            label_path = self.info.loc[i, f'labels_{self.args.label_name}']
            split = self.info.loc[i, "split"]
            if split == 'train':
                labels = _load_point_data(
                    label_path,
                    swap_xy=False,
                    file_type='csv',
                    label_name=self.args.label_name
                )
                resample_ratios.append(calculate_av_label_distance(labels))

        ratio = np.mean(resample_ratios)
        print(f'Average voxel spacing in train and validation split was: {ratio}')
        return ratio

    def _crop_image(self, subject, labels, crop_labels, crop_buffer):
        ''' Crops the image to the edges of the labels

        Args:
            subject (tio.Subject): The subject with the image to crop
            labels (np.array): The labels
            crop_labels (np.array): The labels used to define the cropping bounding box
            crop_buffer (int): The buffer to add to the crop bounding box

        Returns:
            subject: The subject with a cropped image
            labels: The cropped labels
        '''

        subhead_print('Cropping to edges of crop_labels with buffer')

        # viewr = napari.view_image(subject.img.numpy(), name='image', ndisplay=3)

        # find edges of corneas and rhabdoms, incorporate buffer and make mask
        min_vals = (0, 0, 0)
        max_vals = subject.img.shape[1:]

        bbox = (
            np.maximum(
                np.min(np.floor(crop_labels).astype(int), axis=0) - crop_buffer,
                min_vals
            ),
            np.minimum(
                np.max(np.ceil(crop_labels).astype(int), axis=0) + crop_buffer + 1,
                max_vals
            )
        )

        subject = crop_using_bbox(
            subject,
            bbox,
        )

        labels = update_coords_after_crop(labels, bbox)

        return subject, labels, bbox

    def _resample_image_and_labels(
        self,
        img,
        labels,
        crop_area_labels,
        v,
        resample_crop_labels=False,
        swap_xy=False
    ):
        ''' Resamples the image and labels to the given voxel spacing

        Args:
            img (tio.ScalarImage): The image to resample
            labels (np.array): The labels to resample
            crop_area_labels (np.array): The crop area labels to resample
            v (float): The voxel spacing to resample to
            resample_crop_labels (bool, optional): Whether to resample the crop area
            labels. Defaults to False.
            swap_xy (bool, optional): Whether to swap the x and y axes. Defaults to
            False.

        Returns:
            img: The resampled image
            label_locations: The resampled label locations
            crop_area_locations: The resampled crop area locations
        '''

        subhead_print(f'Resampling the image')

        # some warnings to catch bad resample ratios
        if self.resample_ratio < 0.5:
            warnings.warn(
                f'You are increasing the size of your volume by {1/self.resample_ratio}'
                '. This may cause memory issues if your volume is large. '
                'Consider changing your `v` parameter.'
            )

        if self.resample_ratio > 2:
            warnings.warn(
                f'You are reducing the size of your volume by {1/self.resample_ratio}'
                '. Ensure that you want such a small volume. '
                'Consider changing your `v` parameter.'
            )

        subhead_print(f'Resampling image by ratio: {self.resample_ratio}')
        img = resample_by_ratio(img, self.resample_ratio)
        subhead_print(f'Resampling labels by ratio: {self.resample_ratio}')
        label_locations = np.rint(labels / self.resample_ratio)

        if resample_crop_labels:
            subhead_print(
                f'Resampling crop area labels by ratio: {self.resample_ratio}'
            )
            crop_area_locations = np.rint(crop_area_labels / self.resample_ratio)
        else:
            crop_area_locations = None

        return img, label_locations, crop_area_locations

    def _create_output_filename(self, img, split, v, crop):
        out_label_dir = (
            f"./dataset/{self.data_info_name}/{self.folder_prefix}"
            f"/{split}_labels_{str(v)}/"
        )
        out_image_dir = (
            f"./dataset/{self.data_info_name}/{self.folder_prefix}"
            f"/{split}_images_{str(v)}/"
        )

        # make directories if they don't exist
        Path(out_label_dir).mkdir(parents=True, exist_ok=True)
        Path(out_image_dir).mkdir(parents=True, exist_ok=True)
        # get output filenames
        p = Path(img)
        filename = p.stem
        self.label_out_path = out_label_dir + filename
        self.image_out_path = out_image_dir + filename

        # check if the file has already been created
        file_exists = (
            os.path.isfile(f'{self.image_out_path}-image.nii')
            and (not crop or os.path.isfile(f'{self.image_out_path}-0-image.nii'))
        )
        if file_exists:
            return True

        return False

    def _save_whole(self, subject, labels):
        if self.args.debug:
            viewr = napari.view_image(subject.img.numpy(), name='image', ndisplay=2)
            viewr.add_points(labels, name='labels', size=8, face_color='red')
            input('Press enter to continue')

        im_path = self.image_out_path + '-image.nii'
        print('saving image to ' + im_path)
        subject.img.save(im_path)

        lbl_path = self.label_out_path + f'-{self.args.label_name}.csv'
        print('saving labels to ' + lbl_path)
        np.savetxt(lbl_path, labels, delimiter=",")

        rs_ratio_path = self.label_out_path + '-resample_ratio.txt'
        print('saving resample ratio to ' + rs_ratio_path)
        with open(rs_ratio_path, 'w') as f:
            f.write(str(self.resample_ratio))

        bbox_path = self.label_out_path + '-bbox.csv'
        print('saving crop bbox to ' + bbox_path)
        np.savetxt(bbox_path, self.bbox, delimiter=",")

    def _save_patches(self, subject, labels, image_out_path, patch_size):
        # sample patch from whole volume in a grid pattern with padding if the image is
        # too small
        sampler = tio.GridSampler(
            subject=subject,
            patch_size=patch_size,
            padding_mode=0
        )

        num_patches = len(sampler)

        # and save each patch
        print(f"saving {str(num_patches)} patches...")
        patch_locations = sampler.locations
        for i, patch in tqdm(enumerate(sampler), total=num_patches):
            image_path = f'{self.image_out_path}-{i}-image.nii'
            cornea_path = f'{self.label_out_path}-{i}-{self.args.label_name}.csv'

            patch.img.save(image_path)
            labels_in_patch = crop_3d_coords(labels, patch_locations[i])
            np.savetxt(cornea_path, labels_in_patch, delimiter=",")
            # debug plot
            if self.args.debug:
                viewr = napari.view_image(
                    patch.img.numpy(),
                    name='image patch',
                    ndisplay=3
                )
                viewr.add_points(
                    labels_in_patch, size=1, face_color='green', name='labels in patch'
                )
                viewr.add_points(labels, size=1, face_color='red', name='all labels')
                input("Press enter to continue")
        # run the garbage collector to make sure memory is freed
        gc.collect()

    def _save(self, subject, labels, whole=False):
        if self.args.patch_size in [None, 0] or whole:
            self._save_whole(subject, labels)
        else:
            self._save_patches(
                subject, labels, self.image_out_path, self.args.patch_size
            )
        print('finished saving')

    def _process_single_image(self, i, v, whole, crop):
        ''' Processes a single set of image and labels

        Args:
            i (int): The index of the image to process
            v (float): The voxel spacing to resample to
            whole (bool): Whether to save the whole image instead of patches
            crop (bool): Whether to crop the image to the edges of the labels
        '''

        img_path = self.info.loc[i, "image_file_path"]
        label_path = self.info.loc[i, f'labels_{self.args.label_name}']
        split = self.info.loc[i, "split"]
        swap_xy = False

        head_print('Starting conversion of ' + img_path)

        exists = self._create_output_filename(img_path, split, v, crop)
        if exists:
            print(self.image_out_path + ' has already been created, so skipping')
            print('delete the file if you want to recreate it')
            return None

        img = tio.ScalarImage(img_path, check_nans=True)
        labels = _load_point_data(
            label_path,
            swap_xy,
            file_type='csv',
            label_name=self.args.label_name
        )

        if crop:
            temps = list()
            for crop_area_label_name in self.args.crop_label_names:
                crop_area_label = self.info.loc[i, f'labels_{crop_area_label_name}']
                temps.append(
                    _load_point_data(
                        crop_area_label,
                        swap_xy,
                        file_type='csv',
                        label_name=self.args.label_name
                    )
                )
            crop_area_labels = np.concatenate(
                temps,
                axis=0
            )
        else:
            crop_area_labels = None


        self.resample_ratio = calculate_av_label_distance(labels) / v
        
        subject = tio.Subject(img=img)


        if crop:
            # crop before resampling, so we don't have to resample the whole image if only a small part was labelled.
            # avoids ram overusage issues
            subject, labels, bbox = self._crop_image(
                subject,
                labels,
                crop_area_labels,
                int(self.args.crop_buffer / self.resample_ratio) # convert crop buffer in resampled space to original space
            )
            self.bbox = bbox

        # viewr = napari.view_image(img.numpy(), name='image before crop')

        img, labels, crop_area_labels = self._resample_image_and_labels(
            subject.img,
            labels,
            crop_area_labels,
            v,
            resample_crop_labels=crop,
            swap_xy=swap_xy
        )

        # viewr.add_image(subject.img.numpy(), name='image after crop')
        # viewr.add_image(img.numpy(), name='image after crop and resample')
        # viewr.add_points(labels, name='labels after crop and resample', size=8, face_color='red')
        # input()

        subject = tio.Subject(img=img)

        self._save(subject, labels, whole=whole)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='''
        Generate nicely resampled and formatted images and labels from the datas raw
        format.
        '''
    )

    parser.add_argument(
        'data_source_specifier_path',
        type=str,
        help='''
        The path to your .csv file that specifies where your images and labels are and
        whether it should be used for training or testing.
        This .csv file should have the following columns:
        1. image_file_path (path to the dicom or nifti file used to annotate with mctv)
        2. labels_<YOUR_LABEL_NAME> (path to csv file of the labels you which to detect.
        If you would like to use another set of labels to set the label area for
        cropping, you should have multiple columns. e.g. 'labels_corneas',
        'labels_rhabdoms'
        3. split (containing 'train' or 'test' to say how the data should be split)
            This should be generated randomly, or rerun multiple times as part of a
            k-fold cross validation process.

        Example:
            data_source_specifiers/fiddlercrab_corneas.csv
            or
            data_source_specifiers/paraphronima_rhabdoms.csv
        '''
    )

    parser.add_argument(
        '--voxel_spacings',
        '-v',
        nargs='+',
        type=int,
        help='''
        The voxel spacings you would like between your labels to use to resample your
        dataset.
        This is resample size that gives you the average spacing between labels.
        Can be a single value, multiple values or None (the default). If None, then
        the voxel spacing to resample to will be the average found in the dataset.

        One thing to consider in this decision is that an image with too few voxels may
arting conversion of dataset/raw_images/flammula_20200327_female_left_178_fullres_cropped.nii
        not provide enough information for the model to detect the
        feature, whereas a image with too many voxels may have so much information that
        it cannot be loaded into your computer's memory, or require an unreasonably
        large training time for your model. If you face memory usage issues during
        training or inference, consider reducing the voxel spacing used.

        Example:
            -v 10
            or
            -v 8 10 12
        ''',
        default=[None]
    )

    parser.add_argument(
        '--label_name',
        '-l',
        type=str,
        help='''
        The name of the label you would like to detect.
        Currently, only one label name is possible at a time.
        If you would like support for multiple labels, please open an issue at
        https://www.github.com/jakemanger/dhr

        Example:
            -l corneas
            or
            -l rhabdoms
        ''',
        default='corneas'
    )

    parser.add_argument(
        '--crop_label_names',
        '-cl',
        nargs='+',
        type=str,
        help='''
        The name/s of the label or labels you would like to use to define an area for
        cropping the scan with.

        Example:
            -cl corneas
            or
            -cl rhabdoms
            or
            -cl corneas rhabdoms
        ''',
        default=['corneas', 'rhabdoms']
    )

    parser.add_argument(
        '--patch_size',
        '-p',
        type=int,
        help='''
        The patch size to crop your images into smaller sizes that can be loaded
        efficiently. If <= 0 is supplied then only whole images and labels will be
        generated. Cropped images and labels will be placed at
        'dataset/your_config_name/cropped/', while whole images (useful for inference
        and testing) will be placed at 'dataset/your_config_name/whole/'.

        Example:
            -p 0
            or
            -p 256
        ''',
        default=256
    )

    parser.add_argument(
        '--crop_buffer',
        '-cb',
        type=int,
        help='''
        The buffer to add to the crop area to ensure that the entire label is included
        in the cropped image.

        Example:
            -cb 16
        ''',
        default=16
    )

    parser.add_argument(
        '--debug',
        '-d',
        action='store_true',
        help='''
        Whether to run in debug mode. This will generate interactive plots of the
        images and labels to ensure that they are being processed correctly.
        '''
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    ds = DatasetGenerator(args)
    ds.run()
