import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List
import os


def load_dataframe_to_3d_array(df: pd.DataFrame, shape: tuple) -> np.ndarray:
    """
    Convert a DataFrame back to its original 3D array shape.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with multi-index (subject, region, timestep)
    shape : tuple
        Original shape (subjects, regions, timesteps)
    
    Returns
    -------
    np.ndarray
        3D array with shape (subjects, regions, timesteps)
    """
    # Reshape the flattened values back to 3D
    values = df.iloc[:, 0].values  # Get the values column
    return values.reshape(shape)


def reorder_bold_data(df: pd.DataFrame, original_shape: tuple) -> pd.DataFrame:
    """
    Reorder BOLD data from nonsymmetric to symmetric AAL90 ordering.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with multi-index (subject, region, timestep)
    original_shape : tuple
        Original shape (subjects, regions, timesteps)
    
    Returns
    -------
    pd.DataFrame
        Reordered DataFrame
    """
    # Convert back to 3D array
    data_3d = load_dataframe_to_3d_array(df, original_shape)
    
    # Apply reordering to the regions dimension (axis=1)
    subjects, regions, timesteps = data_3d.shape
    
    if regions != 90:
        print(f"Warning: Expected 90 regions, got {regions}. Skipping reordering.")
        return df
    
    # Define index mapping for AAL90 ordering
    left_indices = np.arange(0, 90, 2)        # [0, 2, ..., 88]
    right_indices = np.arange(1, 90, 2)[::-1] # [89, 87, ..., 1]
    aal_indices = np.concatenate([left_indices, right_indices])
    
    # Reorder the regions dimension
    reordered_3d = data_3d[:, aal_indices, :]
    
    # Convert back to DataFrame
    subjects, regions, timesteps = reordered_3d.shape
    index = pd.MultiIndex.from_product([
        range(subjects), range(regions), range(timesteps)
    ], names=['subject', 'region', 'timestep'])
    
    reordered_df = pd.DataFrame(
        reordered_3d.flatten(), 
        index=index, 
        columns=[df.columns[0]]  # Keep original column name
    )
    
    return reordered_df


def process_bold_files(results_dir: str = "results"):
    """
    Process all BOLD files in results directory and apply reordering.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find all BOLD files
    bold_files = list(results_path.glob("BOLD_*.csv"))
    
    if not bold_files:
        print("No BOLD files found in results directory.")
        return
    
    print(f"Found {len(bold_files)} BOLD files to process:")
    
    for file_path in bold_files:
        print(f"\nProcessing: {file_path.name}")
        
        # Load DataFrame
        df = pd.read_csv(file_path, index_col=[0, 1, 2])
        
        # Get original shape
        subjects = df.index.get_level_values('subject').nunique()
        regions = df.index.get_level_values('region').nunique()
        timesteps = df.index.get_level_values('timestep').nunique()
        original_shape = (subjects, regions, timesteps)
        
        print(f"  Original shape: {original_shape}")
        
        # Apply reordering
        reordered_df = reorder_bold_data(df, original_shape)
        
        # Save reordered data
        output_file = results_path / f"reordered_{file_path.name}"
        reordered_df.to_csv(output_file)
        
        print(f"  Saved reordered data: {output_file}")
        print(f"  Reordering applied to regions dimension")


def main():
    """Main function to process all BOLD files."""
    print("=== BOLD Data Reordering Pipeline ===\n")
    
    try:
        process_bold_files()
        print("\n✅ Reordering completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during reordering: {e}")


if __name__ == "__main__":
    main()
