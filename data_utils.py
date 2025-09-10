# data_utils.py
import pandas as pd
from pathlib import Path
import os

def load_nypd_csvs(path_nypd: Path):
    csv_files = sorted(path_nypd.rglob("*.csv"), key=lambda x: x.parent.name)
    dfs = [pd.read_csv(file, low_memory=False) for file in csv_files]
    return pd.concat(dfs, ignore_index=True)

def filter_crashes(df: pd.DataFrame, keyword: str):
    vehicle_cols = [col for col in df.columns if "VEHICLE TYPE CODE" in col]
    df_filtered = df[
        df[vehicle_cols].apply(lambda row: row.str.upper().eq(keyword).any(), axis=1)
    ].copy()
    df_filtered["CRASH DATETIME"] = pd.to_datetime(df_filtered["CRASH DATE"] + " " + df_filtered["CRASH TIME"])
    return df_filtered

def save_if_missing(df: pd.DataFrame, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        df.to_csv(filepath, index=False)
        print(f"✅ Saved: {filepath}")
    else:
        print(f"⏩ File already exists: {filepath}")

def merge_monthly_csvs(folder: Path, output_folder: Path):
    output_folder.mkdir(exist_ok=True)
    for subfolder in sorted(folder.iterdir()):
        if subfolder.is_dir() and subfolder.name.endswith("-citibike-tripdata"):
            month_name = subfolder.name.split("-")[0]
            output_file = output_folder / f"{month_name}_merged.csv"

            if not output_file.exists():
                csv_files = sorted(subfolder.glob("*.csv"))
                if csv_files:
                    dfs = [pd.read_csv(file, low_memory=False) for file in csv_files]
                    df_month = pd.concat(dfs, ignore_index=True)
                    df_month.to_csv(output_file, index=False)
                    print(f"✅ Saved: {output_file}")
                else:
                    print(f"⚠ No CSV files found in {subfolder}")
            else:
                print(f"⏩ File already exists: {output_file}")
