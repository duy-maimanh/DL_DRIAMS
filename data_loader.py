import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
CLEAN_CSV = 'all_clean.csv'
tqdm.pandas(desc='Extract medicine data...!')
processed_location = "./processed_data"

def load_spectra(args, row):
    download_location = args.target_dataset
    df = pd.read_csv(f'{download_location}/{row["institute"]}/binned_6000/{row["year"]}/{row["code"]}.txt', sep=" ")
    return df['binned_intensity'].to_numpy()

def merge_dataset(args):
    download_location = args.target_dataset
    print("Loading data...!")
    all_clean_csv_path = f"{processed_location}/{CLEAN_CSV}"
    if os.path.isfile(all_clean_csv_path) == True:
        print("Found merged data")
        return True
    print("Merging dataset...")
    master_df = pd.DataFrame()

    institutes = [x for x in os.listdir(download_location) 
                  if os.path.isdir(f'{download_location}/{x}') and not x.startswith('.')]
    
    for institute in institutes:
        years = [x for x in os.listdir(f'{download_location}/{institute}/id/') 
                 if os.path.isdir(f'{download_location}/{institute}/id/{x}') and not x.startswith('.')]
        for year in years:
            print(f'{institute} in {year}')
            df = pd.read_csv(f'{download_location}/{institute}/id/{year}/{year}_clean.csv', dtype='string')
            df['year'] = year
            df['institute'] = institute
            if master_df.empty:
                master_df = df
            else:
                master_df = pd.concat([master_df, df], ignore_index=True, sort=False)
    
    # Sort columns based on missing values
    master_df = master_df[master_df.isna().sum().sort_values().keys()]
    master_df.to_csv(all_clean_csv_path, index=False)
    print("Finish merging...!")
    return True

def get_dataset(args):
    medicine = args.medicine
    merge_dataset(args)
    all_clean_csv_path = f"{processed_location}/{CLEAN_CSV}"
    print(all_clean_csv_path)

    pickle_data = f"{processed_location}/{args.medicine}.pkl"
    if os.path.isfile(pickle_data) == True:
        print("Found medicine data")
        pa_df = pd.read_pickle(pickle_data)
    else:
        df = pd.read_csv(all_clean_csv_path, dtype='string', na_values=['-'])
        df=df[df.isna().sum().sort_values().keys()]
        # Drop columns with all NaN values
        pa_df = df.dropna(axis=1, how='all')
    
        # Replace 'I' with 'R' in the antimicrobial column
        pa_df.loc[:, medicine] = pa_df[medicine].replace('I', 'R')
        # Filter rows where the antimicrobial column has 'S' (Susceptible) or 'R' (Resistant)
        pa_df = pa_df[(pa_df[medicine] == 'S') | (pa_df[medicine] == 'R')]
    
        # Select relevant columns for training
        pa_df = pa_df[['code', 'species', medicine, 'year', 'institute']]
    
        pa_df['bins'] = pa_df.progress_apply(lambda x: load_spectra(args, x), axis=1)
        pa_df.to_pickle(pickle_data)

    inputs = np.vstack(pa_df['bins'].to_numpy())
    targets = pa_df[medicine].apply(lambda x: x == 'R').to_numpy()

    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    get_dataset()