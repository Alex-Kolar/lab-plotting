import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/Bulk_crystal/10mK/02072025/PREINIT.csv")
START_FREQ = 194808.612  # unit: GHz
END_FREQ = 194817.320  # unit: GHz
PUMP_START = 194814.488  # unit: GHz
PUMP_END = 194815.680  # unit: GHz
AOM_OFFSET = 0.680  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_od = 'coral'

SHOW_PUMP = True


START_FREQ += AOM_OFFSET
END_FREQ += AOM_OFFSET

df = pd.read_csv(DATA, header=10, skiprows=[11])

ramp = df['CH1'].astype(float).to_numpy()
transmission = df['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

# convert to optical depth
bg = max(transmission)
od = np.log(bg / transmission)

# convert time to frequency
freq = np.linspace(0, (END_FREQ - START_FREQ), id_max-id_min)  # unit: GHz


# plot transmission
plt.plot(freq, transmission, color=color, label="Transmission")
plt.xlabel(f"Frequency + {START_FREQ:.3f} (GHz)")
plt.ylabel("Transmission (A.U.)")
plt.xlim((2.5, 6.5))

plt.tight_layout()
plt.show()


# plot optical depth
plt.plot(freq, od, color=color_od, label="Data")
if SHOW_PUMP:
    pump_1 = PUMP_START - START_FREQ
    pump_2 = PUMP_END - START_FREQ
    print(f"Scan range: {START_FREQ + pump_1:.3f} to {START_FREQ + pump_2:.3f} GHz")
    plt.fill_between(freq, 0, 1, label="Pumping Region",
                     where=np.logical_and(freq >= pump_1, freq <= pump_2),
                     color='gray', alpha=0.2, transform=plt.gca().get_xaxis_transform())
    plt.legend(shadow=True, loc='upper left')

plt.xlabel(f"Frequency + {START_FREQ:.3f} (GHz)")
plt.ylabel("Optical Depth")
plt.xlim((2.5, 6.5))

plt.tight_layout()
plt.show()
