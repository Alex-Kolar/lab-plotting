import numpy


test_file = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
             '/Mounted_device_mk_5/10mK/2026_03_19/afc/afc_storage_experiment.npz')

data = numpy.load(test_file)
counts = data['counts']
print(sum(counts))
