import numpy as np
import pandas as pd
import os
import glob
import argparse
from scipy.spatial.distance import cdist

def load_points_from_csv(csv_path):
    """Load points from CSV file, handling empty files."""
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 5:
        try:
            points = np.loadtxt(csv_path, delimiter=',', skiprows=1, ndmin=2)
            if points.size == 0:
                return np.array([]).reshape(0, 3)
            return points
        except:
            return np.array([]).reshape(0, 3)
    return np.array([]).reshape(0, 3)

def save_points_to_csv(points, csv_path):
    """Save points to CSV file with header."""
    np.savetxt(csv_path, points, delimiter=',', header='x,y,z', comments='')

def find_duplicate_tps(tps, distance_threshold=1.0):
    """Find pairs of TPs that are within distance_threshold of each other."""
    if len(tps) < 2:
        return []
    
    # Calculate pairwise distances
    distances = cdist(tps, tps)
    
    # Find pairs within threshold (excluding self-distances)
    duplicate_pairs = []
    for i in range(len(tps)):
        for j in range(i + 1, len(tps)):
            if distances[i, j] <= distance_threshold:
                duplicate_pairs.append((i, j, distances[i, j]))
    
    return duplicate_pairs

def remove_duplicate_tps(tps, fps, distance_threshold=1.0, removal_strategy='higher_index'):
    """
    Remove duplicate TPs by converting one of each pair to FP.
    
    Args:
        tps: Array of True Positive coordinates
        fps: Array of False Positive coordinates  
        distance_threshold: Maximum distance to consider duplicates
        removal_strategy: 'higher_index', 'lower_index', 'further_from_origin'
    
    Returns:
        updated_tps, updated_fps, changes_log
    """
    if len(tps) < 2:
        return tps, fps, []
    
    duplicate_pairs = find_duplicate_tps(tps, distance_threshold)
    
    if not duplicate_pairs:
        return tps, fps, []
    
    # Determine which points to remove (convert to FP)
    points_to_remove = set()
    changes_log = []
    
    for i, j, distance in duplicate_pairs:
        # Skip if either point is already marked for removal
        if i in points_to_remove or j in points_to_remove:
            continue
            
        # Decide which point to remove based on strategy
        if removal_strategy == 'higher_index':
            remove_idx = j  # Remove the one with higher index
            keep_idx = i
        elif removal_strategy == 'lower_index':
            remove_idx = i  # Remove the one with lower index
            keep_idx = j
        elif removal_strategy == 'further_from_origin':
            # Remove the one further from origin
            dist_i = np.linalg.norm(tps[i])
            dist_j = np.linalg.norm(tps[j])
            if dist_i > dist_j:
                remove_idx = i
                keep_idx = j
            else:
                remove_idx = j
                keep_idx = i
        else:
            # Default to higher index
            remove_idx = j
            keep_idx = i
        
        points_to_remove.add(remove_idx)
        changes_log.append({
            'action': 'TP_to_FP',
            'removed_tp_index': remove_idx,
            'kept_tp_index': keep_idx,
            'removed_coordinates': tps[remove_idx].tolist(),
            'kept_coordinates': tps[keep_idx].tolist(),
            'distance': distance,
            'strategy': removal_strategy
        })
    
    # Create updated arrays
    new_tps = []
    points_to_add_to_fps = []
    
    for idx, tp in enumerate(tps):
        if idx in points_to_remove:
            points_to_add_to_fps.append(tp)
        else:
            new_tps.append(tp)
    
    # Update TPs and FPs
    updated_tps = np.array(new_tps).reshape(-1, 3) if new_tps else np.array([]).reshape(0, 3)
    
    if len(points_to_add_to_fps) > 0:
        if len(fps) > 0:
            updated_fps = np.vstack([fps, np.array(points_to_add_to_fps)])
        else:
            updated_fps = np.array(points_to_add_to_fps)
    else:
        updated_fps = fps
    
    return updated_tps, updated_fps, changes_log

def process_corrected_model(short_filename, distance_threshold=1.0, removal_strategy='higher_index'):
    """Process a single corrected model to remove duplicate TPs."""
    print(f"\n{'='*60}")
    print(f"Processing: {short_filename}")
    print(f"{'='*60}")
    
    # Load current corrected points
    tps_path = f'corrected_results/{short_filename}_cleaned_tps.csv'
    fps_path = f'corrected_results/{short_filename}_cleaned_fps.csv'
    fns_path = f'corrected_results/{short_filename}_cleaned_fns.csv'
    
    if not all(os.path.exists(path) for path in [tps_path, fps_path, fns_path]):
        print(f"Warning: Missing corrected files for {short_filename}")
        return None
    
    tps = load_points_from_csv(tps_path)
    fps = load_points_from_csv(fps_path)
    fns = load_points_from_csv(fns_path)
    
    print(f"Original counts - TPs: {len(tps)}, FPs: {len(fps)}, FNs: {len(fns)}")
    
    # Find and remove duplicate TPs
    updated_tps, updated_fps, changes_log = remove_duplicate_tps(
        tps, fps, distance_threshold, removal_strategy
    )
    
    print(f"Updated counts  - TPs: {len(updated_tps)}, FPs: {len(updated_fps)}, FNs: {len(fns)}")
    
    if changes_log:
        print(f"\nChanges made:")
        for i, change in enumerate(changes_log, 1):
            print(f"  {i}. Converted TP to FP:")
            print(f"     Removed TP at {change['removed_coordinates']} (index {change['removed_tp_index']})")
            print(f"     Kept TP at    {change['kept_coordinates']} (index {change['kept_tp_index']})")
            print(f"     Distance: {change['distance']:.3f} pixels")
            print(f"     Strategy: {change['strategy']}")
        
        # Save updated results
        save_points_to_csv(updated_tps, tps_path)
        save_points_to_csv(updated_fps, fps_path)
        # FNs don't change, no need to save
        
        print(f"\nUpdated files saved successfully!")
        
        # Calculate metric changes
        original_precision = len(tps) / (len(tps) + len(fps)) if (len(tps) + len(fps)) > 0 else 0
        original_recall = len(tps) / (len(tps) + len(fns)) if (len(tps) + len(fns)) > 0 else 0
        original_f1 = 2 * (original_precision * original_recall) / (original_precision + original_recall) if (original_precision + original_recall) > 0 else 0
        
        updated_precision = len(updated_tps) / (len(updated_tps) + len(updated_fps)) if (len(updated_tps) + len(updated_fps)) > 0 else 0
        updated_recall = len(updated_tps) / (len(updated_tps) + len(fns)) if (len(updated_tps) + len(fns)) > 0 else 0
        updated_f1 = 2 * (updated_precision * updated_recall) / (updated_precision + updated_recall) if (updated_precision + updated_recall) > 0 else 0
        
        print(f"\nMetric Changes:")
        print(f"  Precision: {original_precision:.3f} → {updated_precision:.3f} ({updated_precision - original_precision:+.3f})")
        print(f"  Recall:    {original_recall:.3f} → {updated_recall:.3f} ({updated_recall - original_recall:+.3f})")
        print(f"  F1 Score:  {original_f1:.3f} → {updated_f1:.3f} ({updated_f1 - original_f1:+.3f})")
        
        return {
            'short_filename': short_filename,
            'duplicates_found': len(changes_log),
            'original_metrics': {'precision': original_precision, 'recall': original_recall, 'f1': original_f1},
            'updated_metrics': {'precision': updated_precision, 'recall': updated_recall, 'f1': updated_f1},
            'changes': changes_log
        }
    else:
        print("No duplicate TPs found within threshold distance.")
        return None

def find_all_corrected_models():
    """Find all available corrected models."""
    if not os.path.exists('corrected_results'):
        print("No corrected_results directory found.")
        return []
    
    # Find all correction files
    correction_files = glob.glob('corrected_results/*_cleaned_tps.csv')
    available_models = []
    
    for file in correction_files:
        short_filename = os.path.basename(file).replace('_cleaned_tps.csv', '')
        available_models.append(short_filename)
    
    return available_models

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate True Positives by converting close ones to False Positives')
    parser.add_argument('--distance_threshold', type=float, default=1.0,
                       help='Maximum distance (in pixels) to consider TPs as duplicates (default: 1.0)')
    parser.add_argument('--strategy', type=str, default='higher_index',
                       choices=['higher_index', 'lower_index', 'further_from_origin'],
                       help='Strategy for choosing which TP to remove (default: higher_index)')
    parser.add_argument('--model', type=str, default=None,
                       help='Process only this specific model (short_filename)')
    
    args = parser.parse_args()
    
    print(f"Duplicate TP Cleaner")
    print(f"Distance threshold: {args.distance_threshold} pixels")
    print(f"Removal strategy: {args.strategy}")
    
    # Find available models
    available_models = find_all_corrected_models()
    
    if not available_models:
        print("No corrected models found. Run the main correction workflow first.")
        return
    
    if args.model:
        if args.model in available_models:
            models_to_process = [args.model]
        else:
            print(f"Model '{args.model}' not found. Available models:")
            for model in available_models:
                print(f"  {model}")
            return
    else:
        models_to_process = available_models
    
    print(f"\nFound {len(models_to_process)} models to process")
    
    # Process each model
    results = []
    total_duplicates = 0
    
    for model in models_to_process:
        result = process_corrected_model(model, args.distance_threshold, args.strategy)
        if result:
            results.append(result)
            total_duplicates += result['duplicates_found']
    
    # Summary
    print(f"\n{'='*60}")
    print("DUPLICATE CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"Models processed: {len(models_to_process)}")
    print(f"Models with duplicates: {len(results)}")
    print(f"Total duplicates removed: {total_duplicates}")
    
    if results:
        avg_f1_change = np.mean([r['updated_metrics']['f1'] - r['original_metrics']['f1'] for r in results])
        print(f"Average F1 score change: {avg_f1_change:+.3f}")
        
        print(f"\nDetailed Results:")
        print(f"{'Model':<40} {'Duplicates':<12} {'F1 Change':<12}")
        print(f"{'-'*64}")
        for result in results:
            f1_change = result['updated_metrics']['f1'] - result['original_metrics']['f1']
            model_name = result['short_filename'][:37] + "..." if len(result['short_filename']) > 40 else result['short_filename']
            print(f"{model_name:<40} {result['duplicates_found']:<12} {f1_change:<+12.3f}")

if __name__ == "__main__":
    main() 