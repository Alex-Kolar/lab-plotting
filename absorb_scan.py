import glob
import os
import pickle
import matplotlib.pyplot as plt


DATA_DIR = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Lithium Niobate/Absorption Scan/Chip 4"


pickle_files = glob.glob('*.pkl', recursive=True, root_dir=DATA_DIR)
pickle_paths = [os.path.join(DATA_DIR, file) for file in pickle_files]

dicts = []
for file in pickle_paths:
    with open(file, 'rb') as fh:
        dicts.append(pickle.load(fh))

print("Available keys (from pickle files):")
for key in dicts[0]:
    print(f"\t{key}")
print(dicts[0]['Experiment Setup'])


for res in dicts:
    plt.plot(res['Laser wavelength'], res['Power'])

# plt.xlim((1529, 1535))
plt.xlabel("Wavelength (nm)")
plt.ylabel(r"Power ($\mu$W)")
plt.grid('on')

plt.tight_layout()
plt.show()
