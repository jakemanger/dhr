#!/usr/bin/env python3
import os
import glob
import nibabel as nib
import numpy as np
from pathlib import Path

def get_nifti_volume_stats(file_path):
    """Get volume statistics for a NIfTI file."""
    try:
        img = nib.load(file_path)
        shape = img.shape
        voxel_dims = img.header.get_zooms()[:3]
        voxel_volume = np.prod(voxel_dims)
        total_voxels = np.prod(shape[:3])
        total_volume_mm3 = total_voxels * voxel_volume
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        return {
            'shape': shape[:3],
            'voxel_dims': voxel_dims,
            'voxel_volume_mm3': voxel_volume,
            'total_voxels': total_voxels,
            'total_volume_mm3': total_volume_mm3,
            'file_size_mb': file_size_mb
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def analyze_dataset(dataset_name, base_path="/Users/jakemanger/projects/dhr/dataset"):
    """Analyze all NIfTI files in a dataset."""
    dataset_path = Path(base_path) / dataset_name
    
    results = {
        'whole': []
    }
    
    # Analyze whole data only - look in train_images_10 and test_images_10
    whole_path = dataset_path / "whole"
    if whole_path.exists():
        # Check train_images_10
        train_path = whole_path / "train_images_10"
        if train_path.exists():
            train_images = list(train_path.glob("*.nii"))
            for img_file in train_images:
                stats = get_nifti_volume_stats(img_file)
                if stats:
                    stats['filename'] = img_file.name
                    stats['split'] = 'train'
                    results['whole'].append(stats)
        
        # Check test_images_10
        test_path = whole_path / "test_images_10"
        if test_path.exists():
            test_images = list(test_path.glob("*.nii"))
            for img_file in test_images:
                stats = get_nifti_volume_stats(img_file)
                if stats:
                    stats['filename'] = img_file.name
                    stats['split'] = 'test'
                    results['whole'].append(stats)
    
    return results

def print_dataset_summary(dataset_name, results):
    """Print a formatted summary of dataset statistics."""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Separate train and test data
    train_data = [item for item in results['whole'] if item.get('split') == 'train']
    test_data = [item for item in results['whole'] if item.get('split') == 'test']
    
    # Process train images
    if train_data:
        print(f"\n{'TRAIN SET':-^90}")
        print(f"\nTrain Images ({len(train_data)} files):")
        print(f"{'File':<45} {'Dimensions (X,Y,Z)':<20} {'Voxels':<15} {'Size (MB)':<10}")
        print("-" * 90)
        
        train_voxels = []
        train_file_sizes = []
        train_dims_x = []
        train_dims_y = []
        train_dims_z = []
        
        for item in train_data:
            dims_str = f"({item['shape'][0]},{item['shape'][1]},{item['shape'][2]})"
            print(f"{item['filename']:<45} {dims_str:<20} {int(item['total_voxels']):<15} {item['file_size_mb']:<10.2f}")
            train_voxels.append(item['total_voxels'])
            train_file_sizes.append(item['file_size_mb'])
            train_dims_x.append(item['shape'][0])
            train_dims_y.append(item['shape'][1])
            train_dims_z.append(item['shape'][2])
        
        print(f"\nTrain Set Statistics:")
        print(f"  Average voxels: {int(np.mean(train_voxels)):,}")
        print(f"  X dimension: avg={int(np.mean(train_dims_x))}, min={int(np.min(train_dims_x))}, max={int(np.max(train_dims_x))}")
        print(f"  Y dimension: avg={int(np.mean(train_dims_y))}, min={int(np.min(train_dims_y))}, max={int(np.max(train_dims_y))}")
        print(f"  Z dimension: avg={int(np.mean(train_dims_z))}, min={int(np.min(train_dims_z))}, max={int(np.max(train_dims_z))}")
        print(f"  Average file size: {np.mean(train_file_sizes):.2f} MB")
    
    # Process test images
    if test_data:
        print(f"\n{'TEST SET':-^90}")
        print(f"\nTest Images ({len(test_data)} files):")
        print(f"{'File':<45} {'Dimensions (X,Y,Z)':<20} {'Voxels':<15} {'Size (MB)':<10}")
        print("-" * 90)
        
        test_voxels = []
        test_file_sizes = []
        test_dims_x = []
        test_dims_y = []
        test_dims_z = []
        
        for item in test_data:
            dims_str = f"({item['shape'][0]},{item['shape'][1]},{item['shape'][2]})"
            print(f"{item['filename']:<45} {dims_str:<20} {int(item['total_voxels']):<15} {item['file_size_mb']:<10.2f}")
            test_voxels.append(item['total_voxels'])
            test_file_sizes.append(item['file_size_mb'])
            test_dims_x.append(item['shape'][0])
            test_dims_y.append(item['shape'][1])
            test_dims_z.append(item['shape'][2])
        
        print(f"\nTest Set Statistics:")
        print(f"  Average voxels: {int(np.mean(test_voxels)):,}")
        print(f"  X dimension: avg={int(np.mean(test_dims_x))}, min={int(np.min(test_dims_x))}, max={int(np.max(test_dims_x))}")
        print(f"  Y dimension: avg={int(np.mean(test_dims_y))}, min={int(np.min(test_dims_y))}, max={int(np.max(test_dims_y))}")
        print(f"  Z dimension: avg={int(np.mean(test_dims_z))}, min={int(np.min(test_dims_z))}, max={int(np.max(test_dims_z))}")
        print(f"  Average file size: {np.mean(test_file_sizes):.2f} MB")
    
    # Overall summary
    all_data = results['whole']
    if all_data:
        all_voxels = [item['total_voxels'] for item in all_data]
        all_file_sizes = [item['file_size_mb'] for item in all_data]
        all_dims_x = [item['shape'][0] for item in all_data]
        all_dims_y = [item['shape'][1] for item in all_data]
        all_dims_z = [item['shape'][2] for item in all_data]
        
        print(f"\n{'OVERALL STATISTICS':-^90}")
        print(f"Total files: {len(all_data)} (Train: {len(train_data)}, Test: {len(test_data)})")
        print(f"\nVoxel Statistics:")
        print(f"  Average voxels: {int(np.mean(all_voxels)):,}")
        print(f"  Median voxels: {int(np.median(all_voxels)):,}")
        print(f"  Min voxels: {int(np.min(all_voxels)):,}")
        print(f"  Max voxels: {int(np.max(all_voxels)):,}")
        print(f"  Std dev voxels: {int(np.std(all_voxels)):,}")
        
        print(f"\nDimension Statistics:")
        print(f"  X dimension: avg={int(np.mean(all_dims_x))}, min={int(np.min(all_dims_x))}, max={int(np.max(all_dims_x))}")
        print(f"  Y dimension: avg={int(np.mean(all_dims_y))}, min={int(np.min(all_dims_y))}, max={int(np.max(all_dims_y))}")
        print(f"  Z dimension: avg={int(np.mean(all_dims_z))}, min={int(np.min(all_dims_z))}, max={int(np.max(all_dims_z))}")
        
        print(f"\nFile Size Statistics:")
        print(f"  Average file size: {np.mean(all_file_sizes):.2f} MB")
        print(f"  Total dataset size: {np.sum(all_file_sizes):.2f} MB")
    else:
        print(f"\nNo whole image files found in {dataset_name}")

def main():
    datasets = [
        'paraphronima_corneas',
        'paraphronima_rhabdoms',
        'fiddlercrab_corneas',
        'fiddlercrab_rhabdoms'
    ]
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\nAnalyzing {dataset}...")
        results = analyze_dataset(dataset)
        all_results[dataset] = results
        print_dataset_summary(dataset, results)
    
    # Print overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    
    for dataset in datasets:
        results = all_results[dataset]
        all_voxels = []
        all_file_sizes = []
        
        for item in results['whole']:
            all_voxels.append(item['total_voxels'])
            all_file_sizes.append(item['file_size_mb'])
        
        if all_voxels:
            print(f"\n{dataset}:")
            print(f"  Files: {len(all_voxels)}")
            print(f"  Avg voxels: {int(np.mean(all_voxels)):,}")
            print(f"  Avg file size: {np.mean(all_file_sizes):.2f} MB")
            print(f"  Total size: {np.sum(all_file_sizes):.2f} MB")

if __name__ == "__main__":
    main()