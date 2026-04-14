from __future__ import annotations

import os
import numpy as np
import pandas as pd
import h5py
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ftag.hdf5.h5reader import H5Reader
from ftag.cuts import Cuts

from puma.utils import logger
from puma import Histogram, HistogramPlot

file_path = "/home/xzcapfed/MSci/flavour_contamination/testing_cuts/total_efficiencies.pkl"
data = pd.read_pickle(file_path)

# data = data[data['true_bjet_efficiency'] > 0.97]
data['cut_depth'] = 0.5*(data['x_cut'] * data['y_cut'])
data = data.sort_values(by=["cut_depth"])

print(data)

# plotting efficiencies
true_bjet_efficiencies = data['true_bjet_efficiency']
contaminated_fractions = data['contaminated_fraction']
cut_depth = data['cut_depth']

plt.figure(figsize=(8,6))
plt.plot(cut_depth, true_bjet_efficiencies, '.', label='true bjet efficiency')
plt.plot(cut_depth, contaminated_fractions, '.', label='contaminated fraction')
plt.axhline(y=0.97, color='gray', linestyle='--', alpha=0.7, label="efficiency threshold")
plt.ylabel("Fraction")
plt.xlabel("Cut depth")
plt.legend(loc='best', frameon=False)
# plt.title("Cut metrics as a function of cut depth")
plt.savefig('/home/xzcapfed/MSci/flavour_contamination/testing_cuts/final.png')
