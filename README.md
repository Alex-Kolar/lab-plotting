# Lab Plotting Scripts

This repo contains various scripts I have used for creating plots in the lab. This includes both plotting of data and some fitting using the `lmfit` and `scipy` packages.

**NOTE:** there may be allowances for data reading from different oscilloscopes/instruments, plotting parameters, etc. Please double check the code before using output results.

## File structure

There are several subdirectories, each containing various data. The sub-dicrectories are:
- `eps`: For measurements taken using the SPDC entangled photon source. Largely includes HOM data.
- `er_yvo`:
- `ring_resonators`: For measurements of the in-situ entanglement source project. Contains further subfolders:
  - `alignment`: For viewing data from auto-alignment script; used during cooldown.
  - `cavity_fit`: Fitting of Fano resonances from the resonators.
  - `coincidence`: Coincidence measurements of photons generated using SFWM in resonators.
  - `eom`: Measurements and testing of EOM integration.
  - `PL`: PL measurements taken of the sample through the waveguides/resonator.
- `spectroscopy`: Miscellaneous spectroscopy files, mostly for Er:LiN collaboration.
- `testing`: Miscellaneous testing files for plotting, math, etc.
