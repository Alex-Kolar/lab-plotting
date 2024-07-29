import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# data storage directories
DATA_FILE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex"
             "/Ring Resonators/Alignment Tests/Feedthrough Power/power_data_20240604_163918.csv")
LOG_PATH = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/BF1_logdata"

# temperature log parameters
LOG_FMT = "CH{} T {}.log"  # format: channel number, date (y-m-d)
LOG_HEADER = ["Day", "Timestamp", "Temperature (K)"]
LOG_DATETIME = "Datetime"
LOG_TIME_FMT = "%d-%m-%y-%X"
channels = {"50K": 1,
            "4K": 2,
            "Still": 5}

# plotting params
POWER_SCALE = (r'mW', 1e3)  # label, scaling (relative to uW)
power_color = 'cornflowerblue'
temp_color = ['k', 'k', 'k']
temp_ls = ['-', '--', ':']


# data timestamp function
def add_datetime(df, day_key=LOG_HEADER[0], time_key=LOG_HEADER[1], date_key=LOG_DATETIME):
    date_series = pd.Series()
    for i, row in df.iterrows():
        day = row[day_key]
        time = row[time_key]
        time_str = str.join('-', (day, time))
        date_series[i] = datetime.datetime.strptime(
            time_str, LOG_TIME_FMT)
    df[date_key] = date_series
    return df


# data read and prep
power_df = pd.read_csv(DATA_FILE)

time_series = power_df["Epoch Time (s)"]
seconds_series = time_series - time_series[0]
seconds_series /= 3600  # convert to hours
time_series = time_series.apply(datetime.datetime.fromtimestamp)

power_series = power_df["Power (uW)"]
power_series /= POWER_SCALE[1]  # convert to units (see above)

# determine days to extract temperature data
days = set()
for i, time_point in enumerate(time_series):
    day = time_point.strftime('%y-%m-%d')
    days.add(day)

# read all log files
log_data = {}
for day in days:
    data = {}

    for channel_label, channel in channels.items():
        data_file = os.path.join(LOG_PATH, day,
                                 LOG_FMT.format(channel, day))
        df = pd.read_csv(data_file, names=LOG_HEADER)
        df = add_datetime(df)
        data[channel_label] = df

    log_data[day] = data

# down-sample power data to match temperature data


# plotting of power versus time
fig, ax = plt.subplots()
ax_temp = ax.twinx()

# add power
ax.plot(time_series, power_series,
        color=power_color)

# add temp
for i, stage in enumerate(channels):
    for j, day in enumerate(days):
        df = log_data[day][stage]
        if j == 0:
            ax_temp.plot(df['Datetime'], df['Temperature (K)'],
                         color=temp_color[i], ls=temp_ls[i],
                         label=f"{stage} Temperature")
        else:
            ax_temp.plot(df['Datetime'], df['Temperature (K)'],
                         color=temp_color[i], ls=temp_ls[i])

# ax.set_ylim((0, 0.11))
ax.set_xlim((min(time_series), max(time_series)))

ax.tick_params(axis='y', colors=power_color)
ax.set_title("Power versus Time")
ax.set_xlabel("Time")
ax.set_ylabel(rf"Power ({POWER_SCALE[0]})",
              color=power_color)
ax_temp.set_ylabel("Temperature (K)")
ax.grid(True)
ax_temp.legend(shadow=True, loc='upper left')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H%M'))

plt.gcf().autofmt_xdate()
fig.tight_layout()
fig.show()


# plotting of power versus temperature

