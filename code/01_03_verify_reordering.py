"""
Verification script to check if BOLD data reordering was applied correctly.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def load_dataframe_to_3d_array(df: pd.DataFrame, shape: tuple) -> np.ndarray:
    """Convert DataFrame back to 3D array."""
    values = df.iloc[:, 0].values
    return values.reshape(shape)


def verify_reordering(original_file: str, reordered_file: str):
    """
    Verify that reordering was applied correctly.
    
    Parameters
    ----------
    original_file : str
        Path to original BOLD file
    reordered_file : str
        Path to reordered BOLD file
    """
    print(f"=== Verifying reordering: {original_file} -> {reordered_file} ===\n")
    
    # Load both DataFrames
    df_original = pd.read_csv(original_file, index_col=[0, 1, 2])
    df_reordered = pd.read_csv(reordered_file, index_col=[0, 1, 2])
    
    # Get shapes
    subjects_orig = df_original.index.get_level_values('subject').nunique()
    regions_orig = df_original.index.get_level_values('region').nunique()
    timesteps_orig = df_original.index.get_level_values('timestep').nunique()
    
    subjects_reord = df_reordered.index.get_level_values('subject').nunique()
    regions_reord = df_reordered.index.get_level_values('region').nunique()
    timesteps_reord = df_reordered.index.get_level_values('timestep').nunique()
    
    print(f"Original shape: ({subjects_orig}, {regions_orig}, {timesteps_orig})")
    print(f"Reordered shape: ({subjects_reord}, {regions_reord}, {timesteps_reord})")
    
    # Check if shapes match
    if (subjects_orig, regions_orig, timesteps_orig) != (subjects_reord, regions_reord, timesteps_reord):
        print("❌ Shape mismatch!")
        return False
    
    print("✅ Shapes match")
    
    # Convert to 3D arrays
    original_3d = load_dataframe_to_3d_array(df_original, (subjects_orig, regions_orig, timesteps_orig))
    reordered_3d = load_dataframe_to_3d_array(df_reordered, (subjects_reord, regions_reord, timesteps_reord))
    
    # Check if data values are preserved (just reordered)
    original_values = np.sort(original_3d.flatten())
    reordered_values = np.sort(reordered_3d.flatten())
    
    if np.allclose(original_values, reordered_values):
        print("✅ All data values preserved")
    else:
        print("❌ Data values changed!")
        return False
    
    # Verify the specific reordering pattern
    if regions_orig == 90:
        print("\n=== Checking AAL90 reordering pattern ===")
        
        # Define expected reordering indices
        left_indices = np.arange(0, 90, 2)        # [0, 2, ..., 88]
        right_indices = np.arange(1, 90, 2)[::-1] # [89, 87, ..., 1]
        expected_aal_indices = np.concatenate([left_indices, right_indices])
        
        # Check a few specific regions to verify reordering
        test_cases = [
            (0, 0),   # First region should move to position 0
            (1, 89),  # Second region should move to position 89
            (2, 1),   # Third region should move to position 1
            (88, 44), # 89th region should move to position 44
            (89, 45)  # 90th region should move to position 45
        ]
        
        print("Checking specific region mappings:")
        for orig_pos, expected_pos in test_cases:
            if orig_pos < regions_orig:
                # Get a sample of data from this region
                orig_data = original_3d[0, orig_pos, :10]  # First subject, first 10 timesteps
                reord_data = reordered_3d[0, expected_pos, :10]
                
                if np.allclose(orig_data, reord_data):
                    print(f"  ✅ Region {orig_pos} -> {expected_pos}: Correct")
                else:
                    print(f"  ❌ Region {orig_pos} -> {expected_pos}: Incorrect")
                    return False
        
        print("✅ AAL90 reordering pattern verified")
    
    return True


def visualize_reordering(original_file: str, reordered_file: str, subject_idx: int = 0):
    """
    Create visualization to compare original vs reordered data.
    
    Parameters
    ----------
    original_file : str
        Path to original BOLD file
    reordered_file : str
        Path to reordered BOLD file
    subject_idx : int
        Subject index to visualize
    """
    print(f"\n=== Creating visualization for subject {subject_idx} ===")
    
    # Load data
    df_original = pd.read_csv(original_file, index_col=[0, 1, 2])
    df_reordered = pd.read_csv(reordered_file, index_col=[0, 1, 2])
    
    # Get shapes
    subjects = df_original.index.get_level_values('subject').nunique()
    regions = df_original.index.get_level_values('region').nunique()
    timesteps = df_original.index.get_level_values('timestep').nunique()
    
    # Convert to 3D arrays
    original_3d = load_dataframe_to_3d_array(df_original, (subjects, regions, timesteps))
    reordered_3d = load_dataframe_to_3d_array(df_reordered, (subjects, regions, timesteps))
    
    # Extract data for the specified subject
    orig_subject = original_3d[subject_idx, :, :]  # regions x timesteps
    reord_subject = reordered_3d[subject_idx, :, :]
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data heatmap
    im1 = axes[0].imshow(orig_subject, aspect='auto', cmap='viridis')
    axes[0].set_title('Original Data (Subject {})'.format(subject_idx))
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Regions')
    plt.colorbar(im1, ax=axes[0])
    
    # Reordered data heatmap
    im2 = axes[1].imshow(reord_subject, aspect='auto', cmap='viridis')
    axes[1].set_title('Reordered Data (Subject {})'.format(subject_idx))
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Regions')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    # Create figs directory if it doesn't exist
    figs_dir = Path("results/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    output_file = figs_dir / f"reordering_verification_{Path(original_file).stem}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {output_file}")
    
    plt.show()


def main():
    """Main function to verify all reordered files."""
    print("=== BOLD Reordering Verification ===\n")
    
    results_path = Path("results/")
    
    if not results_path.exists():
        print("Results directory not found!")
        return
    
    # Find all reordered files
    reordered_files = list(results_path.glob("reordered_BOLD_*.csv"))
    
    if not reordered_files:
        print("No reordered BOLD files found!")
        return
    
    print(f"Found {len(reordered_files)} reordered files to verify:")
    
    all_passed = True
    
    for reordered_file in reordered_files:
        # Find corresponding original file
        original_name = reordered_file.name.replace("reordered_", "")
        original_file = results_path / original_name
        
        if not original_file.exists():
            print(f"❌ Original file not found: {original_file}")
            all_passed = False
            continue
        
        # Verify reordering
        if verify_reordering(str(original_file), str(reordered_file)):
            print("✅ Verification passed\n")
            
            # Create visualization for first file
            if all_passed:  # Only for first file to avoid too many plots
                try:
                    visualize_reordering(str(original_file), str(reordered_file))
                except Exception as e:
                    print(f"Warning: Could not create visualization: {e}")
        else:
            print("❌ Verification failed\n")
            all_passed = False
    
    if all_passed:
        print("🎉 All reordered files verified successfully!")
    else:
        print("⚠️  Some files failed verification!")


if __name__ == "__main__":
    main() 