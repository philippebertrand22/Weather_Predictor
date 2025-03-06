#%%
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# Base URL of the folder containing CSV files
base_url = "https://dd.weather.gc.ca/climate/observations/daily/csv/QC/"

file_condition = "2020"

# Columns to keep from the CSV files
columns_to_keep = ["STATION_NAME",
                   "LOCAL_YEAR",
                   "LOCAL_MONTH",
                   "LOCAL_DAY",
                   "MEAN_TEMPERATURE"]  # Adjust as needed

# Get the list of CSV files from the directory webpage
response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all links ending in .csv
csv_links = [base_url + link.get("href") for link in soup.find_all("a") if link.get("href").endswith(".csv")
             and file_condition in link.get("href")]

# Function to download and process a CSV file
def download_and_process(file_url):
    print(f"Processing: {file_url}")
    try:
        df = pd.read_csv(file_url, usecols=columns_to_keep)
        return df
    except Exception as e:
        print(f"Error reading {file_url}: {e}")
        return None

# Use ThreadPoolExecutor to download files concurrently
with ThreadPoolExecutor(max_workers=10) as executor:
    dfs = list(executor.map(download_and_process, csv_links))

# Filter out any None values from failed downloads
dfs = [df for df in dfs if df is not None]

# Combine all DataFrames into one (if needed)
final_df = pd.concat(dfs, ignore_index=True)

# Show the first few rows of the final DataFrame
print(final_df.head())

#%%
