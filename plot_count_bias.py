import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})

df = pd.read_csv('~/Desktop/Projects/FPGA_Control/snspd_count.csv')

fig, ax = plt.subplots()
ax.plot(df['bias'], df['count_9'], '-o', color='cornflowerblue', label='Channel 9')
ax.plot(df['bias'], df['count_10'], '-o', color='coral', label='Channel 10')


ax.set_xlim((5, 18))
ax.set_ylim((-500, 9e3))
ax.grid('on')

ax.set_xlabel(r'Bias Current ($\mu$A)')
ax.set_ylabel(r'Counts per Second')
ax.legend(shadow=True)

plt.tight_layout()
plt.savefig('count_bias')
plt.show()
