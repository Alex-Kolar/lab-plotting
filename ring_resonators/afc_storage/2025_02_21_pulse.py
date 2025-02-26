import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel


DATA_TO_FIT = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/Bulk_crystal/storage pulse/02232025/045NS.csv")
TIME_UNIT = 2  # unit: ns
FIT_REGION = (500, 600)

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 14})
color = 'cornflowerblue'


# collect data
df = pd.read_csv(DATA_TO_FIT, header=10, skiprows=[11])

fungen = df['CH1'].astype(float).to_numpy()
light = df['CH2'].astype(float).to_numpy()

time = np.linspace(0, TIME_UNIT*len(light), len(light))

# fit data
fit_idx = np.where(np.logical_and(time >= FIT_REGION[0], time < FIT_REGION[1]))[0]
time_to_fit = time[fit_idx]
light_to_fit = light[fit_idx]

model = GaussianModel() + ConstantModel()
res = model.fit(light_to_fit, x=time_to_fit,
                center=550, sigma=45, c=0)
print(res.fit_report())


fig, ax = plt.subplots(figsize=(8, 6))

plt.plot(time, light, color=color,
         label='Data')
plt.plot(time_to_fit, res.best_fit,
         color='k', ls='--', label='Fit')

plt.title('Storage Pulse Fit')
plt.xlabel('Time (ns)')
plt.ylabel('Photodiode response (V)')
plt.legend()
plt.xlim(FIT_REGION)

fig.tight_layout()
plt.show()
