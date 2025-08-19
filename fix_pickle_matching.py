#!/usr/bin/env python3
"""
Fix pickle file matching by reading contents directly instead of relying on filename patterns.
"""

import pandas as pd
import pickle
import os
import glob
from pathlib import Path

def load_pickle_contents_map():
    """
    Create a mapping from y_hat_path to pickle file by reading pickle contents directly.
    This is much more reliable than filename-based matching.
    """
    print("Loading pickle files and creating content-based mapping...")
    
    pickle_files = glob.glob("./analysis_output/*.pickle")
    y_hat_to_pickle_map = {}
    
    for pickle_file in pickle_files:
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            # Each pickle contains a list of results, get the y_hat_path from each item
            for item in data:
                if 'y_hat_path' in item:
                    y_hat_path = item['y_hat_path']
                    y_hat_to_pickle_map[y_hat_path] = pickle_file
                    print(f"Mapped: {os.path.basename(y_hat_path)} -> {os.path.basename(pickle_file)}")
                    
        except Exception as e:
            print(f"Error reading {pickle_file}: {e}")
            continue
    
    return y_hat_to_pickle_map

def fix_pickle_column_in_dataframe(df):
    """
    Fix the pickle column in the DataFrame by using content-based mapping.
    """
    print("\nFixing pickle column in DataFrame...")
    
    # Create the content-based mapping
    y_hat_to_pickle_map = load_pickle_contents_map()
    
    # Function to map y_hat_path to pickle file using content-based lookup
    def map_y_hat_to_pickle(y_hat_path):
        return y_hat_to_pickle_map.get(y_hat_path, None)
    
    # Apply the mapping
    df['pickle'] = df['y_hat_path'].apply(map_y_hat_to_pickle)
    
    # Show results
    mapped_count = df['pickle'].notna().sum()
    total_count = len(df)
    print(f"\nResults: {mapped_count}/{total_count} rows successfully mapped to pickle files")
    
    # Show which ones couldn't be mapped
    unmapped = df[df['pickle'].isna()]
    if len(unmapped) > 0:
        print(f"\nUnmapped y_hat_paths:")
        for y_hat_path in unmapped['y_hat_path'].values:
            print(f"  {y_hat_path}")
    
    return df

def main():
    # Load the current DataFrame
    print("Loading best_models.csv...")
    df = pd.read_csv('best_models.csv')
    
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Original pickle column - mapped: {df['pickle'].notna().sum()}, unmapped: {df['pickle'].isna().sum()}")
    
    # Fix the pickle column
    df_fixed = fix_pickle_column_in_dataframe(df)
    
    # Save the fixed DataFrame
    df_fixed.to_csv('best_models_fixed.csv', index=False)
    print(f"\nSaved fixed DataFrame to best_models_fixed.csv")
    
    print(f"Fixed pickle column - mapped: {df_fixed['pickle'].notna().sum()}, unmapped: {df_fixed['pickle'].isna().sum()}")
    
    return df_fixed

if __name__ == "__main__":
    df_fixed = main()