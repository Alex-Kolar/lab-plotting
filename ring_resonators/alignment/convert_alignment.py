import os
import pandas as pd
from time import strftime, localtime


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex"
             "/Ring Resonators/Alignment Tests/")
DATA_FILE = "position_data_20240521_113952.csv"
DATE_STR = '%Y-%m-%d %H:%M:%S'

data_path = os.path.join(DATA_DIR, DATA_FILE)
df = pd.read_csv(data_path)

# convert epoch column to time
time_series = df["Epoch Time (s)"]
timestamps = []
for time in time_series:
    datetime = localtime(int(time))  # time must be passed as int
    time_str = strftime(DATE_STR, datetime)
    timestamps.append(time_str)

# save data
df["Timestamp"] = timestamps
# save_name = DATA_FILE[:-4]  # remove '.csv'
# save_name += "_mod.csv"
save_path = os.path.join(DATA_DIR, DATA_FILE)

df.to_csv(save_path)
