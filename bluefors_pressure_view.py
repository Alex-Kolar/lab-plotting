import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


LOG_PATH = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/BF1_logdata"
DAYS_TO_VIEW = ["24-09-27"]

# pressure log parameters
LOG_FMT = "CH{} P {}.log"  # format: channel number, date (y-m-d)
LOG_HEADER = ["Day", "Timestamp", "Pressure (B)"]
LOG_DATETIME = "Datetime"
LOG_TIME_FMT = "%d-%m-%y-%X"
channels = {"Channel 3": 3}

# plotting params (NOTE: same order as 'channels' above)
temp_color = ['cornflowerblue']
temp_ls = '-'
temp_marker = ''
y_limits = [(0, 1)]
share_x = True


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


# read all log files
log_data = {}
for day in DAYS_TO_VIEW:
    data = {}

    for channel_label, channel in channels.items():
        data_file = os.path.join(LOG_PATH, day,
                                 LOG_FMT.format(channel, day))
        df = pd.read_csv(data_file, names=LOG_HEADER)
        df = add_datetime(df)
        data[channel_label] = df

    log_data[day] = data


# plotting of temp versus time
if share_x:
    fig, axs = plt.subplots(len(channels), 1,
                            sharex=True, figsize=(8, 8))

    for i, stage in enumerate(channels):
        for j, day in enumerate(DAYS_TO_VIEW):
            df = log_data[day][stage]
            axs[i].plot(df['Datetime'], df['Temperature (K)'],
                        color=temp_color[i], ls=temp_ls, marker=temp_marker)
            axs[i].set_ylabel(f"{stage}")
            axs[i].set_ylim(y_limits[i])
            axs[i].grid(True, axis='x')

    axs[0].set_title(f"Temperature versus Time")
    fig.supylabel("Temperature (K)")
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H%M'))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.show()

else:
    for i, stage in enumerate(channels):
        fig, ax = plt.subplots()

        for j, day in enumerate(DAYS_TO_VIEW):
            df = log_data[day][stage]
            ax.plot(df['Datetime'], df['Temperature (K)'],
                    color=temp_color[i], ls=temp_ls, marker=temp_marker)

        ax.set_title(f"{stage} Temperature versus Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature (K)")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H%M'))
        plt.grid(True)

        ax.set_ylim(y_limits[i])

        # ax.set_xlim(datetime.datetime(2024, 10, 30),
        #             datetime.datetime(2024, 12, 31))

        plt.gcf().autofmt_xdate()
        fig.tight_layout()
        fig.show()
