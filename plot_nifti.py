#!/usr/bin/env python3
"""
Simple script to load and visualize NIfTI files with napari
"""

import napari
import nibabel as nib
import torchio as tio
import argparse
import numpy as np


def load_nifti_with_nibabel(file_path):
    """
    Load a NIfTI file using nibabel and return as numpy array
    
    Args:
        file_path (str): Path to the NIfTI file
        
    Returns:
        np.ndarray: Image data as numpy array
    """
    img = nib.load(file_path)
    return img.get_fdata()


def load_nifti_with_torchio(file_path):
    """
    Load a NIfTI file using TorchIO ScalarImage
    
    Args:
        file_path (str): Path to the NIfTI file
        
    Returns:
        tio.ScalarImage: TorchIO image object
    """
    img = tio.ScalarImage(file_path, check_nans=True)
    return img


def visualize_nifti(file_path, use_torchio=True):
    """
    Load and visualize a NIfTI file with napari
    
    Args:
        file_path (str): Path to the NIfTI file
        use_torchio (bool): Whether to use TorchIO (True) or nibabel (False)
    """
    
    if use_torchio:
        # Load with TorchIO (following your project's pattern)
        img = load_nifti_with_torchio(file_path)
        image_data = img.numpy()
        print(f"Image shape: {img.shape}")
        print(f"Image spacing: {img.spacing}")
        print(f"Image orientation: {img.orientation}")
    else:
        # Load with nibabel
        image_data = load_nifti_with_nibabel(file_path)
        print(f"Image shape: {image_data.shape}")
    
    # Create napari viewer and display image
    viewer = napari.view_image(
        image_data, 
        name=f'NIfTI: {file_path.split("/")[-1]}',
        contrast_limits=None  # Auto-adjust contrast
    )
    
    print(f"Loaded and displayed: {file_path}")
    print("Use napari viewer to explore the 3D volume")
    
    # Start napari event loop
    napari.run()


def main():
    parser = argparse.ArgumentParser(
        description='Load and visualize NIfTI files with napari'
    )
    parser.add_argument(
        'nifti_path', 
        type=str, 
        help='Path to the NIfTI file (.nii or .nii.gz)'
    )
    parser.add_argument(
        '--use-nibabel', 
        action='store_true',
        help='Use nibabel instead of TorchIO for loading'
    )
    
    args = parser.parse_args()
    
    # Visualize the NIfTI file
    visualize_nifti(args.nifti_path, use_torchio=not args.use_nibabel)


if __name__ == "__main__":
    main()