import numpy as np
from deep_radiologist.lazy_heatmap import LazyHeatmapReader
import torchio as tio
import argparse
import napari


def load_nifti_image(file_path):
    img = tio.ScalarImage(
        file_path,
        check_nans=True
    )
    return img


def load_coords(file_path, img):
    locations = np.loadtxt(
        file_path,
        delimiter=',',
        ndmin=2,
        dtype=np.float64
    )

    smpl_map_reader = LazyHeatmapReader(
        affine=img.affine,
        start_shape=img.shape,
        voxel_size=20
    )
    smpl_map = tio.Image(
        path=file_path,
        type=tio.LABEL,
        check_nans=True,
        reader=smpl_map_reader.read,
    )
    return locations, smpl_map


def main():
    parser = argparse.ArgumentParser(
        description='Napari tool for image and coords checking'
    )
    parser.add_argument('image_path', type=str, help='Path to the NIfTI image file.')
    parser.add_argument('coords_path', type=str, help='Path to the CSV file containing coordinates.')
#     parser.add_argument('csv_file', type=str, help='Path to the csv file with dataset')
    args = parser.parse_args()

    image = load_nifti_image(args.image_path)

    transform = tio.ToCanonical()
    image = transform(image)

    coords, coords_image = load_coords(args.coords_path, image)

    viewer = napari.Viewer()
    viewer.add_image(image.data.numpy(), name='NIfTI Image')
    viewer.add_image(coords_image.data.numpy(), name='NIfTI Image')
    viewer.add_points(coords, size=5, face_color='pink', name='Coords')

    napari.run()

    print('saving')
    coords_image.save(f'{args.coords_path}visualisation.nii')


if __name__ == '__main__':
    main()
