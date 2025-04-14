# src/data/loader.py
"""
Handles loading raw datasets (SGCC, Ausgrid) from source files.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Optional

def load_sgcc_data(
    filepath: str,
    na_threshold_ratio: float = 0.6,
    perform_basic_cleaning: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the SGCC dataset from a CSV file, performs basic cleaning,
    and separates features from theft labels.

    Args:
        filepath (str): The path to the SGCC CSV file.
        na_threshold_ratio (float): Maximum allowed ratio of NA values per row.
                                   Rows exceeding this threshold are dropped.
        perform_basic_cleaning (bool): If True, applies initial NA filling
                                      and row removal based on na_threshold_ratio.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - DataFrame with time series consumption data (time as columns, consumers as rows).
            - Series with the theft flag (1 for theft, 0 for benign) indexed by consumer ID.

    Raises:
        FileNotFoundError: If the filepath does not exist.
        ValueError: If essential columns ('CONS_NO', 'FLAG') are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")

    data = pd.read_csv(filepath)

    if 'CONS_NO' not in data.columns or 'FLAG' not in data.columns:
        raise ValueError("CSV file must contain 'CONS_NO' and 'FLAG' columns.")

    if perform_basic_cleaning:
        # Basic NA handling: fill forward limit 1, then check row-wise NA ratio
        data = data.fillna(method='ffill', limit=1)
        data['na_percentage'] = data.isna().sum(axis=1) / data.shape[1]
        data = data[data['na_percentage'] <= na_threshold_ratio]
        data = data.drop(columns=['na_percentage'])
        # Fill remaining NAs with 0 - consider if other strategies are better
        data = data.fillna(0)

    # Set consumer number as index
    if 'CONS_NO' in data.columns:
        data = data.set_index('CONS_NO')

    # Separate labels
    if 'FLAG' in data.columns:
        thief_labels = data['FLAG'].astype(int)
        data = data.drop(columns=['FLAG'])
    else:
        # Handle case where FLAG might already be separated or not present
        # Depending on requirements, might raise an error or return None for labels
        thief_labels = pd.Series(dtype=int) # Return empty series if no FLAG

    # Convert columns (dates) to datetime objects if possible and sort
    try:
        # Attempt conversion assuming columns are date strings
        date_columns = pd.to_datetime(data.columns, errors='coerce')
        # Filter out columns that couldn't be converted
        valid_date_columns = data.columns[~date_columns.isna()]
        # Keep only valid date columns and sort
        data = data[valid_date_columns]
        data.columns = pd.to_datetime(data.columns) # Ensure they are datetime objects
        data = data.sort_index(axis=1)
    except Exception:
        print("Warning: Could not convert all columns to datetime. Ensure columns represent time steps.")
        # If columns are not dates, maybe just ensure they are sorted if that makes sense
        data = data.sort_index(axis=1)


    # It seems the original code expected time as columns, consumers as rows.
    # Let's keep it that way for consistency with the original Loader.
    # If the standard format is time-as-rows, consumers-as-columns, transpose here:
    # data = data.T
    # data.index = pd.to_datetime(data.index)
    # data.index.name = 'Timestamp'

    return data, thief_labels


def _prepare_ausgrid_dataframe(df: pd.DataFrame, date_format_code: int = 1) -> pd.DataFrame:
    """
    Helper function to prepare a single Ausgrid year dataframe.
    Melts, converts datetime, and pivots the table.

    Args:
        df (pd.DataFrame): Raw Ausgrid dataframe for a single year.
        date_format_code (int): 0 for '%d-%b-%y %H:%M', 1 for '%d/%m/%Y %H:%M'.

    Returns:
        pd.DataFrame: Processed dataframe with datetime index and customers as columns.
    """
    # Filter for 'GC' and drop unnecessary columns
    df = df[df['Consumption Category'] == 'GC'].drop(
        columns=['Consumption Category', 'Generator Capacity', 'Postcode'], errors='ignore'
    )
    if 'Row Quality' in df.columns:
        df = df.drop(columns=['Row Quality'])

    # Melt dataframe to long format
    df_melt = df.melt(id_vars=['Customer', 'date'], var_name='time', value_name='value')

    # Define date formats
    formats = {
        0: '%d-%b-%y %H:%M',
        1: '%d/%m/%Y %H:%M'
    }
    date_format = formats.get(date_format_code)
    if not date_format:
         raise ValueError(f"Invalid date_format_code: {date_format_code}. Use 0 or 1.")

    # Combine 'date' and 'time' columns into datetime
    # Using errors='coerce' to handle potential parsing issues gracefully
    df_melt['datetime'] = pd.to_datetime(
        df_melt['date'] + ' ' + df_melt['time'], format=date_format, errors='coerce'
    )

    # Drop rows where datetime conversion failed
    df_melt = df_melt.dropna(subset=['datetime'])

    # Sort values by 'Customer' and 'datetime'
    df_melt = df_melt.sort_values(by=['Customer', 'datetime'])

    # Drop original 'date' and 'time' columns
    df_melt = df_melt.drop(columns=['date', 'time'])

    # Pivot the table: datetime becomes index, Customer becomes columns
    df_pivot = df_melt.pivot(index='datetime', columns='Customer', values='value')

    # Sort by the datetime index
    df_pivot = df_pivot.sort_index()

    return df_pivot


def load_ausgrid_data(
    dir_paths: List[str],
    filenames: List[str] = ['ausgrid2010.csv', 'ausgrid2011.csv', 'ausgrid2012.csv'],
    date_formats: List[int] = [0, 1, 1] # 0 for 2010, 1 for 2011, 1 for 2012
) -> pd.DataFrame:
    """
    Loads and combines Ausgrid data from multiple CSV files.

    Args:
        dir_paths (List[str]): List of directory paths containing the Ausgrid CSV files.
                               Assumes filenames correspond to these directories in order.
        filenames (List[str]): List of CSV filenames (e.g., ['ausgrid2010.csv', ...]).
        date_formats (List[int]): List of date format codes (0 or 1) corresponding
                                 to each file/year.

    Returns:
        pd.DataFrame: Combined and preprocessed Ausgrid dataframe with datetime index,
                      customers as columns, and forward-filled NAs.

    Raises:
        FileNotFoundError: If any specified CSV file is not found.
        ValueError: If lists dir_paths, filenames, and date_formats have different lengths.
    """
    if not (len(dir_paths) == len(filenames) == len(date_formats)):
        raise ValueError("Lists dir_paths, filenames, and date_formats must have the same length.")

    all_dfs = []
    for i, dir_path in enumerate(dir_paths):
        filepath = os.path.join(dir_path, filenames[i])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ausgrid file not found: {filepath}")

        # Load raw CSV, skipping the header row often present in these files
        raw_df = pd.read_csv(filepath, skiprows=[0])

        # Prepare the dataframe using the helper function
        prepared_df = _prepare_ausgrid_dataframe(raw_df, date_formats[i])
        all_dfs.append(prepared_df)

    # Concatenate dataframes vertically (along rows/time axis)
    ausgrid_combined_df = pd.concat(all_dfs, axis=0)

    # Forward fill missing values - check if this is the desired strategy
    ausgrid_combined_df = ausgrid_combined_df.ffill()

    # Optional: backward fill any remaining NAs at the beginning
    ausgrid_combined_df = ausgrid_combined_df.bfill()

    return ausgrid_combined_df


# Example Usage (commented out)
# if __name__ == "__main__":
#     # SGCC Example
#     sgcc_data_path = r"D:\evaluateAttacks\data.csv" # Adjust path as needed
#     try:
#         sgcc_df, sgcc_labels = load_sgcc_data(sgcc_data_path)
#         print("SGCC Data Loaded:")
#         print(sgcc_df.head())
#         print("\nSGCC Labels:")
#         print(sgcc_labels.head())
#         print(f"\nShape: {sgcc_df.shape}, Labels count: {len(sgcc_labels)}")
#         print(f"Theft count: {sgcc_labels.sum()}")
#     except FileNotFoundError as e:
#         print(e)
#     except ValueError as e:
#         print(e)

#     # Ausgrid Example
#     ausgrid_dirs = [
#         r'path\to\ausgrid2010', # Adjust paths as needed
#         r'path\to\ausgrid2011',
#         r'path\to\ausgrid2012'
#     ]
#     try:
#         ausgrid_df = load_ausgrid_data(ausgrid_dirs)
#         print("\nAusgrid Data Loaded:")
#         print(ausgrid_df.head())
#         print(f"\nShape: {ausgrid_df.shape}")
#         print(f"NA count after ffill: {ausgrid_df.isna().sum().sum()}")
#     except FileNotFoundError as e:
#         print(e)
#     except ValueError as e:
#         print(e)
