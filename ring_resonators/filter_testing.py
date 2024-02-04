import pandas as pd
import matplotlib.pyplot as plt


LASER_REF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data"
             "/Ring Resonators/filter_data/DWDM_CH_50/initial_scan_12152023/laser_ref.csv")
FILTER = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data"
          "/Ring Resonators/filter_data/DWDM_CH_50/initial_scan_12152023/filter_out.csv")
COLD_50K = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data"
            "/Ring Resonators/filter_data/DWDM_CH_50/cooldown_12182023/50K.csv")
COLD_3K = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data"
           "/Ring Resonators/filter_data/DWDM_CH_50/cooldown_12182023/3K.csv")
OSA_HEADER = ["Wavelength (nm)", "Power (dBm)"]

laser_ref_df = pd.read_csv(LASER_REF, names=OSA_HEADER)
filter = pd.read_csv(FILTER, names=OSA_HEADER)
filter_50K = pd.read_csv(COLD_50K, names=OSA_HEADER)
filter_3K = pd.read_csv(COLD_3K, names=OSA_HEADER)

wl = laser_ref_df["Wavelength (nm)"]
filter_suppress_room_temp = filter["Power (dBm)"] - laser_ref_df["Power (dBm)"]
filter_suppress_50K = filter_50K["Power (dBm)"] - laser_ref_df["Power (dBm)"]
filter_suppress_3K = filter_3K["Power (dBm)"] - laser_ref_df["Power (dBm)"]

plt.plot(wl, filter_suppress_room_temp, color='coral')
plt.ylim((-40, 0))
plt.title("Room Temperature")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission (dB)")
plt.grid(True)
plt.show()

plt.plot(wl, filter_suppress_50K, color='cornflowerblue')
plt.ylim((-40, 0))
plt.title("50K Temperature")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission (dB)")
plt.grid(True)
plt.show()

plt.plot(wl, filter_suppress_3K, color='mediumpurple')
plt.ylim((-40, 0))
plt.title("3K Temperature")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission (dB)")
plt.grid(True)
plt.show()

# plot all 3 combined
plt.plot(wl, filter_suppress_room_temp,
         color='coral', label="Room Temperature")
plt.plot(wl, filter_suppress_50K,
         color='cornflowerblue', label="50 K")
plt.plot(wl, filter_suppress_3K,
         color='mediumpurple', label="3 K")
plt.ylim((-40, 0))
plt.title("Filter Cooldown")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission (dB)")
plt.legend(shadow=True)
plt.grid(True)
plt.show()
