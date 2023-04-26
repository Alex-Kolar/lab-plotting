import matplotlib.pyplot as plt
import numpy as np


spectrum = np.arange(1530, 1540.1, 0.5)
signal = [140, 150, 140, 180, 170, 190, 210, 220, 300, 350, 260, 210, 200, 230, 180, 150, 150, 120, 160, 130, 150]
idler = [220, 200, 230, 250, 230, 280, 270, 320, 350, 370, 360, 300, 300, 260, 230, 200, 220, 180, 200, 200, 210]

assert len(signal) == len(idler) == len(spectrum)

plt.plot(spectrum, signal)
plt.plot(spectrum, idler)

plt.xlabel("Wavelength (nm)")
plt.ylabel("(Counts/s)")
plt.legend(["Signal", "Idler"])
plt.savefig("spectrum")
