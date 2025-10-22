import os
import glob


DATA_DIR = '/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er2O3'


csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
npy_files = glob.glob(os.path.join(DATA_DIR, '*.npy'))

for npy_file in npy_files:
    base = os.path.basename(npy_file)
    if '-' in base:
        new_base = base.replace('-', '_')
        os.rename(npy_file, os.path.join(DATA_DIR, new_base))
