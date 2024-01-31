## convert raw label json files to dataframe csv

import os
import json
import pandas as pd
import numpy as np

"""
convert raw json label files to dataframe csv file

Notice: Description data in raw label won't be used. Descriptions will be written by us

Params:
--folder_path
"""
########################################################################

def convert_json_to_csv(folder_path):
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
    df_list = []

    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            data = pd.json_normalize(data)
        df = pd.DataFrame(data)
        new_data = []
        for item in df['images']:
            if item:
                new_data.append(item[0])
        new_df = pd.DataFrame(new_data)
        df_list.append(new_df)

    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        print(merged_df.head())
    else:
        print("no valid json files found.")

    return merged_df

def save_label_df(df):
    data = df
    data.to_csv('./extracted_label_from_raw.csv', encoding='utf-8', index=False)


# folder_path = './label_raw'
# merged_df = convert_json_to_csv(folder_path)
#
# save_label_df(merged_df)


