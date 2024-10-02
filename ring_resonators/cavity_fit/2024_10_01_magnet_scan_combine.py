import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


DATA_FILE = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
             "/new_mounted/10mK_magnet_scan/10mK_10012024/linear_bg/all_scans/res_data.bin")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


with open(DATA_FILE, "rb") as f:
    data = pickle.load(f)

mag_fields = np.array(data['Magnetic Field'])
qs_1 = np.array(data['q_1'])
qs_2 = np.array(data['q_2'])

# exclude data from the highest field
idx_to_exclude = np.where(mag_fields >= 990)
mag_fields = np.delete(mag_fields, idx_to_exclude)
qs_1 = np.delete(qs_1, idx_to_exclude)
qs_2 = np.delete(qs_2, idx_to_exclude)

# do averaging
unique_b = np.unique(mag_fields)
q1_avgs = []
q1_errs = []
q2_avgs = []
q2_errs = []
for b in unique_b:
    idx = np.where(mag_fields == b)
    q1_avg = np.mean(qs_1[idx])
    q1_err = np.std(qs_1[idx])
    q2_avg = np.mean(qs_2[idx])
    q2_err = np.std(qs_2[idx])
    q1_avgs.append(q1_avg)
    q1_errs.append(q1_err)
    q2_avgs.append(q2_avg)
    q2_errs.append(q2_err)
q1_avgs = np.array(q1_avgs)
q1_errs = np.array(q1_errs)
q2_avgs = np.array(q2_avgs)
q2_errs = np.array(q2_errs)

plt.errorbar(unique_b, q1_avgs/1e6, yerr=q1_errs/1e6,
             ls='', marker='o', color='cornflowerblue', capsize=3)
plt.title("Low Q Resonance")
plt.xlabel("Magnetic Field (mT)")
plt.ylabel(r"Fitted Q ($\times 10^6)$")

plt.tight_layout()
plt.show()

plt.errorbar(unique_b, q2_avgs/1e6, yerr=q2_errs/1e6,
             ls='', marker='o', color='coral', capsize=3)
plt.title("High Q Resonance")
plt.xlabel("Magnetic Field (mT)")
plt.ylabel(r"Fitted Q ($\times 10^6)$")

plt.tight_layout()
plt.show()
