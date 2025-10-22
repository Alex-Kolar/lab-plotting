import pickle
import matplotlib.pyplot as plt


DATA_FILE = '/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er2O3/processed_data.bin'


with open(DATA_FILE, 'rb') as f:
    data = pickle.load(f)


for field, field_data in data.items():
    print(f'processing field {field*1e3} mT')

    for freq, volt in zip(field_data['frequencies'], field_data['voltages']):
        plt.plot(freq, volt, ls='', marker=',', color='tab:blue')

    plt.title(f'{field*1e3} mT Field')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Voltage (V)')
    plt.xlim(193970, 194030)
    plt.tight_layout()
    plt.show()
