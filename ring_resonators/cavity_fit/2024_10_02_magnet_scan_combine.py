import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


DATA_FILE = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
             "/new_mounted/10mK_magnet_scan/10mK_10022024/constant_bg/all_scans/res_data.bin")

DATA_TO_EXCLUDE = {
    760: [0],
    820: [9],
    850: [0],
    870: [0],
    890: [0],
    900: [0],
    910: [0],
    960: [0],
    990: [9],
    720: [0],
}

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


with open(DATA_FILE, "rb") as f:
    data = pickle.load(f)

mag_fields = np.array(data['Magnetic Field'])
qs_1 = data['q_1']
qs_2 = data['q_2']

# exclude some data that includes edges of scan
for i, field in enumerate(mag_fields):
    if field in DATA_TO_EXCLUDE:
        for j in DATA_TO_EXCLUDE[field]:
            del qs_1[i][j]
            del qs_2[i][j]

# do averaging
q1_avgs = np.array([np.mean(qs) for qs in qs_1])
q1_errs = np.array([np.std(qs) for qs in qs_1])
q2_avgs = np.array([np.mean(qs) for qs in qs_2])
q2_errs = np.array([np.std(qs) for qs in qs_2])

plt.errorbar(mag_fields, q1_avgs/1e6, yerr=q1_errs/1e6,
             ls='', marker='o', color='cornflowerblue', capsize=3)
plt.title("Low Q Resonance")
plt.xlabel("Magnetic Field (mT)")
plt.ylabel(r"Fitted Q ($\times 10^6)$")

plt.tight_layout()
plt.show()

plt.errorbar(mag_fields, q2_avgs/1e6, yerr=q2_errs/1e6,
             ls='', marker='o', color='coral', capsize=3)
plt.title("High Q Resonance")
plt.xlabel("Magnetic Field (mT)")
plt.ylabel(r"Fitted Q ($\times 10^6)$")

plt.tight_layout()
plt.show()
