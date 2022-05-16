from deep_radiologist.lightning_modules import DataModule, Model
import multiprocessing
import os
import torch
import argparse
import yaml
from yaml.loader import SafeLoader
import napari


def main():
    parser = argparse.ArgumentParser(
        description=''
    )

    parser.add_argument(
        'config_path',
        type=str,
        help='''
        The path to the config file.

        Example:
            configs/fiddlercrab_cornea_config.yaml
        '''
    )

    parser.add_argument(
        '--n_plot_at_once',
        type=int,
        default=4,
        help='''
        The number of images to plot at once.
        '''
    )

    args = parser.parse_args()

    # load config
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)

    data = DataModule(config=config)
    data.prepare_data()
    data.setup()

    model = Model(
        config=config
    )

    n_plotted = 0
    plotted_first_n = False
    for i, subjects in enumerate([data.subjects, data.test_subjects]):
        for subject in subjects:
            print(f'Viewing {subject.filename}')

            if not plotted_first_n:
                viewer = napari.view_image(subject.image.numpy(), name=f'{subject.filename} image')
                plotted_first_n = True
            else:
                viewer.add_image(subject.image.numpy(), name=f'{subject.filename} image')
            viewer.add_image(subject.label.numpy(), name=f'{subject.filename} label')

            coords = model._locate_coords(subject.label.numpy())
            breakpoint()
            viewer.add_points(coords, name=f'{subject.filename} coords', size=4, face_color='blue')

            n_plotted += 1

            if n_plotted == args.n_plot_at_once:
                n_plotted = 0
                input('Press enter to continue')
                viewer.layers.clear()


if __name__ == '__main__':
    main()
