"""
DoC Data Loader and Converter for Brain-Act Project

Loads structural connectivity (subjects x regions x regions) and fMRI BOLD timeseries 
(subjects x regions x time-steps) data, converts to pandas DataFrames, and saves to results/.
"""

import numpy as np
import pandas as pd
from scipy import io
from pathlib import Path
import os
import warnings


class DocDataProcessor:
    """Process DoC data and convert to DataFrames."""
    
    def __init__(self, data_dir="data", results_dir="results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(exist_ok=True)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load_mat_file(self, filepath):
        """Load MATLAB file and return contents."""
        print(f"Loading: {filepath}")
        mat_contents = io.loadmat(str(filepath))
        # Remove MATLAB metadata
        return {k: v for k, v in mat_contents.items() if not k.startswith('__')}
    
    def convert_3d_to_dataframe(self, data, name, dim_names=None):
        """Convert 3D array to DataFrame with proper indexing."""
        if dim_names is None:
            dim_names = ['subject', 'dim1', 'dim2']
        
        # Create multi-index
        subjects, dim1, dim2 = data.shape
        index = pd.MultiIndex.from_product([
            range(subjects), range(dim1), range(dim2)
        ], names=dim_names)
        
        # Flatten and create DataFrame
        df = pd.DataFrame(data.flatten(), index=index, columns=[f'{name}_value'])
        return df
    
    def process_structural_connectivity(self):
        """Process structural connectivity data."""
        print("=== Processing Structural Connectivity ===")
        
        sc_dir = self.data_dir / "DoC_SC"
        if not sc_dir.exists():
            print(f"SC directory not found: {sc_dir}")
            return {}
        
        sc_dataframes = {}
        
        for mat_file in sc_dir.glob("*.mat"):
            print(f"\nProcessing: {mat_file.name}")
            mat_contents = self.load_mat_file(mat_file)
            
            for var_name, data in mat_contents.items():
                if isinstance(data, np.ndarray) and data.ndim == 3:
                    print(f"  Converting {var_name}: shape {data.shape}")
                    
                    # Convert to DataFrame
                    df = self.convert_3d_to_dataframe(
                        data, var_name, ['subject', 'region1', 'region2']
                    )
                    
                    # Save to results
                    output_file = self.results_dir / f"SC_{var_name}.csv"
                    df.to_csv(output_file)
                    print(f"  Saved: {output_file}")
                    
                    sc_dataframes[var_name] = df
                    
                    # Print summary
                    n_subjects, n_regions, _ = data.shape
                    print(f"    Subjects: {n_subjects}, Regions: {n_regions}")
        
        return sc_dataframes
    
    def process_bold_timeseries(self):
        """Process BOLD timeseries data."""
        print("\n=== Processing BOLD Timeseries ===")
        
        bold_dir = self.data_dir / "DoC_bold_correct_indicies"
        if not bold_dir.exists():
            print(f"BOLD directory not found: {bold_dir}")
            return {}
        
        bold_dataframes = {}
        
        for mat_file in bold_dir.glob("*.mat"):
            print(f"\nProcessing: {mat_file.name}")
            mat_contents = self.load_mat_file(mat_file)
            
            for var_name, data in mat_contents.items():
                if isinstance(data, np.ndarray) and data.ndim == 3:
                    print(f"  Converting {var_name}: shape {data.shape}")
                    
                    # Convert to DataFrame
                    df = self.convert_3d_to_dataframe(
                        data, var_name, ['subject', 'region', 'timestep']
                    )
                    
                    # Save to results
                    output_file = self.results_dir / f"BOLD_{var_name}.csv"
                    df.to_csv(output_file)
                    print(f"  Saved: {output_file}")
                    
                    bold_dataframes[var_name] = df
                    
                    # Print summary
                    n_subjects, n_regions, n_timesteps = data.shape
                    print(f"    Subjects: {n_subjects}, Regions: {n_regions}, Timesteps: {n_timesteps}")
        
        return bold_dataframes
    
    def create_summary_report(self, sc_dataframes, bold_dataframes):
        """Create a summary report of all processed data."""
        print("\n=== Creating Summary Report ===")
        
        summary_data = []
        
        # SC data summary
        for name, df in sc_dataframes.items():
            # Get original shape from multi-index
            subjects = df.index.get_level_values('subject').nunique()
            regions = df.index.get_level_values('region1').nunique()
            
            summary_data.append({
                'Data_Type': 'Structural_Connectivity',
                'Variable': name,
                'Subjects': subjects,
                'Regions': regions,
                'Shape': f"{subjects}x{regions}x{regions}",
                'File_Size_MB': os.path.getsize(self.results_dir / f"SC_{name}.csv") / (1024*1024)
            })
        
        # BOLD data summary
        for name, df in bold_dataframes.items():
            subjects = df.index.get_level_values('subject').nunique()
            regions = df.index.get_level_values('region').nunique()
            timesteps = df.index.get_level_values('timestep').nunique()
            
            summary_data.append({
                'Data_Type': 'BOLD_Timeseries',
                'Variable': name,
                'Subjects': subjects,
                'Regions': regions,
                'Timesteps': timesteps,
                'Shape': f"{subjects}x{regions}x{timesteps}",
                'File_Size_MB': os.path.getsize(self.results_dir / f"BOLD_{name}.csv") / (1024*1024)
            })
        
        # Create and save summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / "data_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Summary saved: {summary_file}")
        print("\nData Summary:")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def process_all_data(self):
        """Process all DoC data and convert to DataFrames."""
        print("=== DoC Data Processing Pipeline ===\n")
        
        # Process structural connectivity
        sc_dataframes = self.process_structural_connectivity()
        
        # Process BOLD timeseries
        bold_dataframes = self.process_bold_timeseries()
        
        # Create summary report
        summary_df = self.create_summary_report(sc_dataframes, bold_dataframes)
        
        print(f"\n=== Processing Complete ===")
        print(f"Results saved in: {self.results_dir}")
        print(f"Total files created: {len(sc_dataframes) + len(bold_dataframes) + 1}")
        
        return sc_dataframes, bold_dataframes, summary_df


def main():
    """Main function to run the data processing pipeline."""
    try:
        processor = DocDataProcessor()
        sc_dataframes, bold_dataframes, summary_df = processor.process_all_data()
        
        print("\n✅ Data processing completed successfully!")
        return sc_dataframes, bold_dataframes, summary_df
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return None, None, None


if __name__ == "__main__":
    sc_dataframes, bold_dataframes, summary_df = main() 