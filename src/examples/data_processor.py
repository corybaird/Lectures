# Think of '__init__' as the special setup step when you create a new startup tool.
# It's like the instructions for getting the tool ready to use.

import pandas as pd
import requests
import io
import os

class DataProcessor:
    # '__init__' is a special function that runs automatically when you make a DataProcessor tool.
    def __init__(self, intermediate_folder="intermediate/", raw_folder="raw/"):
        # Inside '__init__', we're giving our new startup tool some initial settings.
        # 'self' here just refers to the specific DataProcessor tool you are creating.
        # We're setting up where this tool will look for and store its information.
        self.intermediate_folder = intermediate_folder  # Where we'll put partially finished info.
        self.raw_folder = raw_folder                    # Where the original info is kept.
        self.df_macro = None                             # Will hold macro info after download.
        self.df_oecd = None                              # Will hold OECD info after download.
        self.df_macro_nz = None                          # Will hold filtered and renamed NZ macro info.
        self.df_oecd_nz = None                           # Will hold filtered and renamed NZ OECD info.
        self.df_merged = None                            # Will hold the combined info.

    # This is a new ability for our tool: downloading macro info from a website.
    def download_macro(self, macro_url):
        print("Running download_macro...")
        self.df_macro = pd.read_stata(macro_url)
        print("Macro data downloaded successfully.")
        return self.df_macro

    # Here's another new ability: downloading info from the OECD website.
    def download_oecd(self, oecd_url):
        print("Running download_oecd...")
        response = requests.get(oecd_url)
        response.raise_for_status()
        data = io.StringIO(response.text)
        self.df_oecd = pd.read_csv(data)
        print("OECD data downloaded successfully.")
        return self.df_oecd

    # This ability filters the macro info to only include data for New Zealand (NZL),
    # selects specific columns, removes any missing data, and renames the columns.
    def filter_rename_macro_nz(self):
        print("Running filter_rename_macro_nz...")
        self.df_macro_nz = self.df_macro.query("ISO3 == 'NZL'")[['ISO3', 'year', 'OECD_KEI_infl', 'BIS_infl']].dropna().copy()
        self.df_macro_nz.rename({"ISO3":'country', "year":'date'}, axis=1, inplace=True)
        print("Macro data filtered and renamed for NZ.")
        return self.df_macro_nz

    # This ability filters the OECD info to only include data for New Zealand (NZL)
    # for a specific measure ('ULCE') and unit ('PA'), selects relevant columns,
    # renames them, and removes unnecessary columns.
    def filter_rename_oecd_nz(self):
        print("Running filter_rename_oecd_nz...")
        cols_oecd = ['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE', 'MEASURE', 'UNIT_MEASURE']
        self.df_oecd_nz = self.df_oecd[cols_oecd].query("REF_AREA == 'NZL' & MEASURE=='ULCE' & UNIT_MEASURE == 'PA'").copy()
        self.df_oecd_nz.rename({"REF_AREA":'country', "TIME_PERIOD":'date', 'OBS_VALUE':'ULCE'}, axis=1, inplace=True)
        self.df_oecd_nz.drop(["MEASURE", "UNIT_MEASURE"], axis=1, inplace=True)
        print("OECD data filtered and renamed for NZ.")
        return self.df_oecd_nz

    # This ability converts the 'date' column in the New Zealand macro info
    # from a year format to a standard date format.
    def convert_datetime_macro_nz(self):
        print("Running convert_datetime_macro_nz...")
        self.df_macro_nz['date'] = pd.to_datetime(self.df_macro_nz['date'], format = '%Y').dt.date
        print("Macro dates converted to datetime.")
        return self.df_macro_nz

    # This ability converts the 'date' column in the New Zealand OECD info
    # from a quarterly format (like '1990-Q4') to a standard date format.
    def convert_datetime_oecd_nz(self):
        print("Running convert_datetime_oecd_nz...")
        self.df_oecd_nz['date'] = pd.PeriodIndex(self.df_oecd_nz['date'], freq='Q').to_timestamp().date
        print("OECD dates converted to datetime.")
        return self.df_oecd_nz

    # This ability sets the 'country' and 'date' columns as the main identifiers (index)
    # for the New Zealand macro info. This helps in combining data later.
    def set_index_macro_nz(self):
        print("Running set_index_macro_nz...")
        self.df_macro_nz.set_index(['country', 'date'], inplace=True)
        print("Macro data index set to country and date.")
        return self.df_macro_nz

    # This ability does the same as above, but for the New Zealand OECD info.
    def set_index_oecd_nz(self):
        print("Running set_index_oecd_nz...")
        self.df_oecd_nz.set_index(['country', 'date'], inplace=True)
        print("OECD data index set to country and date.")
        return self.df_oecd_nz

    # This ability combines the New Zealand macro and OECD info into a single table
    # based on matching 'country' and 'date'. It only keeps the data that exists in both tables.
    def merge_data(self):
        print("Running merge_data...")
        self.df_merged = pd.merge(
            self.df_macro_nz,
            self.df_oecd_nz,
            right_index = True,
            left_index = True,
            how = 'inner'
        )
        print("Macro and OECD data merged.")
        return self.df_merged

    # This ability handles exporting our processed data to files.
    def export_data(self, output_dir_intermediate="data/intermediate/processed/", output_dir_raw="data/raw/"):
        print("Running export_data...")
        os.makedirs(output_dir_intermediate, exist_ok=True)
        merged_filepath = os.path.join(output_dir_intermediate, "merged_data_nz.csv")
        self.df_merged.to_csv(merged_filepath)
        print(f"Merged data exported to: {merged_filepath}")

        os.makedirs(output_dir_raw, exist_ok=True)
        raw_oecd_filepath = os.path.join(output_dir_raw, "oecd_raw.csv")
        self.df_oecd.to_csv(raw_oecd_filepath)
        print(f"Raw OECD data exported to: {raw_oecd_filepath}")