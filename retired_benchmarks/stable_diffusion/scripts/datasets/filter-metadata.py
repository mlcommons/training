import os
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

def filter_parquet_files(input_folder, output_folder):
    # Create output directory if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of all parquet files in each directory
    parquet_files = [f for f in os.listdir(input_folder) if f.endswith('.parquet')]

    # Iterate over all the files in the input folder, filter out rows where LICENSE column has a value of "?"
    for file in parquet_files:
        df = pd.read_parquet(os.path.join(input_folder, file))
        df_filtered = df[df['LICENSE'] != '?']
        
        print(f'{file}: Original samples = {len(df)}, Samples after LICENSE filtering = {len(df_filtered)}')        
        
        output_path = os.path.join(output_folder, file)
        df_filtered.to_parquet(output_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Filter parquet files.')
    parser.add_argument('--input-folder', required=True, help='Path to the input folder.')
    parser.add_argument('--output-folder', required=True, help='Path to the output folder.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    filter_parquet_files(args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()
