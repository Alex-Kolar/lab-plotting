import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


PHASE_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/eom_data/08082024/100 no atten.csv")
AMP_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/eom_data/08092024/100.csv")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


# find and read files
phase_data = pd.read_csv(PHASE_PATH, header=1)
amp_data = pd.read_csv(AMP_PATH, header=1)


plt.plot(phase_data['Freq'] / 1e6, phase_data['Amp'],
         label='Phase Modulation')
plt.plot(amp_data['Freq'] / 1e6, amp_data['Amp'],
         label='Amplitude Modulation')

plt.title(f"Comparing Methods")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dBm)")
plt.legend(shadow=True)
plt.grid(True)

plt.tight_layout()
plt.show()


plt.plot(phase_data['Freq'] / 1e6, phase_data['Amp'],
         color='coral')

plt.title("Laser Burn Pulse")
plt.xlabel("Detuning from Carrier (MHz)")
plt.ylabel("Power (dBm)")
plt.xlim(95, 105)

plt.tight_layout()
plt.show()
