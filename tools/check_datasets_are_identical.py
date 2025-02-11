import logging
from pathlib import Path
import pandas as pd
import csv

if __name__ == "__main__":

    base_dir = Path("/Users/antonellaschiavoni/Documents/Antonella/tesis-ciencia-de-datos/data/processed")
    
    file_old = base_dir / "dataset/dataset_vowel_segmentation_v2_2024_07_14/extracted_features_20240714_064550.csv"
    file_new = base_dir / "datasets/vowel_features/vowel_features_20250209_224102.csv"
    
    df_old = pd.read_csv(file_old)
    df_new = pd.read_csv(file_new)

    are_identical = df_new.equals(df_old)
    logging.info(f"Files are identical: {are_identical}")
    print(f"Files are identical: {are_identical}")

    # print(df_new.columns)
    # print(df_old.columns)

    # print(df_new.describe())
    # print(df_old.describe())

    with open(file_new) as f1, open(file_old) as f2:
        reader_new = csv.reader(f1)
        reader_old = csv.reader(f2)
        
        # Compare headers first
        header_new = next(reader_new)
        header_old = next(reader_old)
        if header_new != header_old:
            print("Headers differ!")
            print(header_new)
            print(header_old)
        
        # Compare remaining rows
        for i, (row_new, row_old) in enumerate(zip(reader_new, reader_old), 1):
            if row_new != row_old:

                print(f"Row {i} differs. \n row_new: \n {row_new} vs \n row_old: \n {row_old}")
        else:
            print("All rows match")

