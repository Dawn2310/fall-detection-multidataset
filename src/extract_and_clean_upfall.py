import os
import argparse
import zipfile
import pandas as pd


def extract_and_clean(base_dir: str = None):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "3D_skeletons-UP-Fall-Dataset-main",
                                "3D_skeletons-UP-Fall-Dataset-main")
    output_dir = os.path.join(base_dir, "extracted_data")
    
    subjects_zips = {
        "SUBJECT1.zip": "subject1",
        "SUBJECT2.zip": "subject10", # standardizing as subject10 because files are S10
        "SUBJECT3.zip": "subject3",
        "SUBJECT4.zip": "subject4",
        "SUBJECT5.zip": "subject7"   # standardizing as subject7 because files are S7
    }
    
    print("Starting extraction and data cleaning...")
    print(f"Source folder: {base_dir}")
    print(f"Destination folder: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for zip_name, folder_name in subjects_zips.items():
        zip_path = os.path.join(base_dir, zip_name)
        if not os.path.exists(zip_path):
            print(f"Error: {zip_name} not found in {base_dir}")
            continue
            
        dest_folder = os.path.join(output_dir, folder_name)
        os.makedirs(dest_folder, exist_ok=True)
        print(f"\nProcessing {zip_name} -> {folder_name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if not member.endswith('.csv'):
                    continue
                
                # Correct some file name typos
                clean_member = member
                if member == "3S3A4T1.csv":
                    clean_member = "C3S3A4T1.csv"
                elif member == "CS3A4T2.csv":
                    clean_member = "C1S3A4T2.csv"
                
                # Extract file data to memory
                with zf.open(member) as f:
                    df = pd.read_csv(f)
                    
                # Fix column headers if needed
                cols = list(df.columns)
                
                # 1. Check for 'LLABEL' typo
                if 'LLABEL' in cols:
                    df.rename(columns={'LLABEL': 'LABEL'}, inplace=True)
                    print(f"  [{member}] Fixed column header 'LLABEL' -> 'LABEL'")
                
                # 2. Check for '0' header typo in Subject 10's file
                if cols[-1] == '0' and len(cols) == 100:
                    df.rename(columns={'0': 'LABEL'}, inplace=True)
                    print(f"  [{member}] Fixed column header '0' -> 'LABEL'")
                
                # Save cleaned CSV file
                out_file_path = os.path.join(dest_folder, clean_member)
                df.to_csv(out_file_path, index=False)
                
    print("\nExtraction and cleaning completed successfully!")
    print(f"Cleaned dataset is saved at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and clean UP-Fall 3D skeleton data.")
    parser.add_argument("--dir", default=None, help="Path to 3D_skeletons folder")
    args = parser.parse_args()
    extract_and_clean(args.dir)
