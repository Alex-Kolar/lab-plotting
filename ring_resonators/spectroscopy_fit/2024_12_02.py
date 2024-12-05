import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/Bulk_crystal/10mK/12022024/absorption.csv")
START_FREQ = 194809.596  # unit: GHz
END_FREQ = 194818.188  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_od = 'coral'


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
plt.xlim((2, 6))

plt.tight_layout()
plt.show()


# plot optical depth
plt.plot(freq, od, color=color_od, label="Optical Depth")
plt.xlabel(f"Frequency + {START_FREQ:.3f} (GHz)")
plt.ylabel("Optical Depth")
plt.xlim((2, 6))

plt.tight_layout()
plt.show()
