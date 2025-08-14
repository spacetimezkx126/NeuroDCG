import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# def read_cmin(directory_path):
def load_price(directory_path):
    """
    Loads all .txt files from the CMIN-CN/price/preprocessed directory.
    """
    stock_data = {}
    label = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            company_name = filename.split('.')[0]
            file_path = os.path.join(directory_path, filename)
            stock_data[company_name] = {}
            label[company_name] = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    time = parts[0]
                    features = [float(x) for x in parts[3:6]]
                    stock_data[filename.replace(".txt","")][time] = features
                    label[filename.replace(".txt","")][time] = parts[1]
    return stock_data, label

def read_cmin(directory_path):
    """
    Reads all JSON files from the cmin_cn_extracted directory structure.

    Args:
        directory_path (str): The path to the cmin_cn_extracted directory.

    Returns:
        dict: A dictionary where keys are company names and values are dictionaries
              of their loaded JSON data, with filenames as keys.
    """
    all_data = {}
    for company_dir in os.listdir(directory_path):
       
        company_path = os.path.join(directory_path, company_dir)
        if os.path.isdir(company_path):
            
            company_data = {}
            count = 0
            count_num = 0
            for json_file in os.listdir(company_path):
                
                if json_file.endswith('.json'):
                    file_path = os.path.join(company_path, json_file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        
                        try:
                            data = json.load(f)
                            if len(data[json_file.replace(".json","")]) != 0 and (not (len(data[json_file.replace(".json","")]) == 1 and len(data[json_file.replace(".json","")][0]) == 0)):
                                company_data[json_file.replace(".json","")] = data
                                # print(len(data[json_file.replace(".json","")][0]))
                                count += len(data[json_file.replace(".json","")][0])
                                count_num += 1
                        except json.JSONDecodeError:
                                print(f"Could not decode JSON from {file_path}")
            all_data[company_dir] = company_data
    return all_data

def read_cmin_us_price(directory_path):
    """
    Reads all .txt files from the CMIN-US/price/processed directory.

    Args:
        directory_path (str): The path to the processed directory.

    Returns:
        dict: A dictionary where keys are tickers and values are pandas DataFrames
              containing the price data.
    """
    all_data = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            ticker = filename.split('.')[0]
            file_path = os.path.join(directory_path, filename)
            try:
                # The file is space-separated, and there's no header.
                # The raw string is used to avoid SyntaxWarning
                df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
                # It looks like the first column is a date. Let's make it the index.
                df = df.set_index(0)
                df.index.name = 'date'
                all_data[ticker] = df
            except Exception as e:
                print(f"Could not read or process file {file_path}: {e}")
    return all_data