"""Produce histograms for calo data"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from puma import Roc, RocPlot
from ftag.utils import calculate_rejection
from puma.utils import logger
import h5py
import pandas as pd
from puma.utils import get_good_colours
from ftag.cuts import Cuts
from ftag.hdf5.h5reader import H5Reader
from ftag import Flavours
import glob

# ------- Functions -------
def load_data(file_pattern, num_jets=None, batch_size=500_000):
    """Loads and filters data from HDF5 files matching a pattern using H5Reader.

    Parameters
    ----------
    file_pattern : str
        Glob pattern (e.g., "/data/*.h5") for HDF5 files
    num_jets : int, optional
        Max number of jets to load
    batch_size : int, optional
        Batch size to read at a time

    Returns
    -------
    pd.DataFrame
        Filtered jet data as a pandas DataFrame
    """
    # Expand wildcard pattern into list of file paths
    file_paths = glob.glob(file_pattern)
    if not file_paths:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    # Define cuts as before
    cuts = Cuts.from_list([
        ("pt", ">", 20000),
        ("pt", "<", 6000000),
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
        # ("eventNumber", "%10==", 9),
    ])

    # Define required variables
    jet_vars = [
        "pt", "eta", "eventNumber",
        "HadronGhostTruthLabelID",
        "flavour_label"
    ]
    calo_vars = [
            "rawE",          
            "PreSamplerB",   
            "PreSamplerE",   
            "EMB1",
            "EMB2",          
            "EME1",  
            "EME2",        
            "HEC0",          
            "TileBar0",            
            "TileExt0",         
        ]

    variables = {"jets": jet_vars,
                 "calo": calo_vars}

    reader = H5Reader(
        fname=file_paths,
        jets_name="jets",
        batch_size=batch_size,
        do_remove_inf=True,
        shuffle=True
    )

    data = reader.load(variables=variables, cuts=cuts, num_jets=num_jets)

    df_jets = pd.DataFrame(data["jets"])

    # (N-jets, 50, 4) so each jet has 50 elements and 4 features
    calo_3d = np.stack([
            data['calo']['rawE'],
            data['calo']['PreSamplerB'],
            data['calo']['PreSamplerE'],
            data['calo']['EMB1'],
            data['calo']['EMB2'],
            data['calo']['EME1'],
            data['calo']['EME2'],
            data['calo']['HEC0'],
            data['calo']['TileBar0'],
            data['calo']['TileExt0'],
            ], axis=-1
        )

    return df_jets, calo_3d

def describe_df(df, column=None):
    """Prints a summary of the given dataframe to the terminal
    
    Parameters:
    df : dataframe table (pd.dataframe)
    column : column to produce statsitics for (str)
    """
    pd.set_option('display.max_columns', None) # show all columns

    print("---SHAPE INFO----")
    print(df.info())
    print(df.describe())

    if column:
        print("---LOOK at desired COLUMN---")
        print(df[column].describe())

    return

def filtered_calo_feature(calo_array, feature_index, mask):
    """
    Extracts a 2D feature from a 3D calorimeter array, applies a boolean mask, 
    flattens the resulting data, and removes zero values.
    
    Parameters
    calo_array : calorimeter data with shape(N_jets, 50, 4) (np.ndarray)
    feature_index : feature to extract e.g., 0 for LONGITUDINAL (int)
    mask : Boolean array to filter the jets (np.ndarray)
    
    Returns:
    1D array of the flattened, non-zero feature values (np.ndarray)
    """
    feature_2d = calo_array[:, :, feature_index]
    flattened_feature = np.nansum(feature_2d[mask], axis=1)
    
    return flattened_feature[flattened_feature != 0.0]

def get_layer_fraction(calo_array, layer_idx, rawE_idx, mask):
    """
    Calculates the fraction of total energy deposited in a specific layer per jet.
    
    Parameters
    calo_array : calorimeter data with shape(N_jets, 50, 4) (np.ndarray)
    layer_idx  : feature index for the specific layer (int)
    rawE_idx   : feature index for the total energy, usually rawE (int)
    mask       : Boolean array to filter the jets by flavour/pt (np.ndarray)
    
    Returns:
    1D array of the energy fractions (np.ndarray)
    """
    layer_energy = np.nansum(calo_array[mask, :, layer_idx], axis=1)
    total_energy = np.nansum(calo_array[mask, :, rawE_idx], axis=1)
    
    valid_jets = total_energy > 0
    
    fractions = layer_energy[valid_jets] / total_energy[valid_jets]
    
    return fractions


# ------- Plotting -------
fname = ""

logger.info("Loading h5 file")
df_jets, calo_3d = load_data(fname)
describe_df(df_jets)
print(calo_3d.shape)

is_ttbar = (df_jets['pt'] > 20000) & (df_jets['pt'] < 250000) 
is_zprime = (df_jets['pt'] > 250000) & (df_jets['pt'] < 6000000)
is_b = df_jets['HadronGhostTruthLabelID'] == 5
is_c = df_jets['HadronGhostTruthLabelID'] == 4
is_tau = df_jets['HadronGhostTruthLabelID'] == 15

# load variables
rawE_b = filtered_calo_feature(calo_3d, 0, is_b)
PresamplerB_b = filtered_calo_feature(calo_3d, 1, is_b)
EMB1_b = filtered_calo_feature(calo_3d, 2, is_b)
HEC0_b = filtered_calo_feature(calo_3d, 3, is_b)

rawE_c = filtered_calo_feature(calo_3d, 0, is_c)
PresamplerB_c = filtered_calo_feature(calo_3d, 1, is_c)
EMB1_c = filtered_calo_feature(calo_3d, 2, is_c)
HEC0_c = filtered_calo_feature(calo_3d, 3, is_c)

rawE_tau = filtered_calo_feature(calo_3d, 0, is_tau)
PresamplerB_tau = filtered_calo_feature(calo_3d, 1, is_tau)
EMB1_tau = filtered_calo_feature(calo_3d, 2, is_tau)
HEC0_tau = filtered_calo_feature(calo_3d, 3, is_tau)

# histograms 
atlas_label = (
    r"$\mathbfit{ATLAS}$ Simulation Preliminary" + "\n"
    r"$\sqrt{s} = 13.6$ TeV" + "\n"
    r" $t\bar{t}$ events, 20 GeV < $p_{T}$ < 250 GeV" + "\n"
    r" $Z'$ events, 250 GeV < $p_{T}$ < 6 TeV"
)


# ------- Calculating Mean Fractions -------
logger.info("Calculating mean energy fractions across all layers...")

def get_mean_fraction(calo_array, layer_idx, rawE_idx, mask):
    """Calculates the mean energy fraction (layer / total) for a filtered set of jets."""
    layer_energy = np.nansum(calo_array[mask, :, layer_idx], axis=1)
    total_energy = np.nansum(calo_array[mask, :, rawE_idx], axis=1)
    
    valid_jets = total_energy > 0
    fractions = layer_energy[valid_jets] / total_energy[valid_jets]
    return np.mean(fractions)

# Define the layers and their exact indices in the calo_3d array
# If you want to see the missing energy, update your dictionary like this!
layers = {
    'PreSamplerB': 1,
    'PreSamplerE': 2,
    'EMB1': 3,
    'EMB2': 4,
    'EME1': 5,
    'EME2' : 6,
    'HEC0': 7,
    'TileBar0': 8,
    'TileExt0': 9
}

rawE_idx = 0

# Lists to store the means for plotting
means_b = []
means_c = []
means_tau = []

# Loop through the dictionary and calculate the mean for each flavour
for name, idx in layers.items():
    means_b.append(get_mean_fraction(calo_3d, layer_idx=idx, rawE_idx=rawE_idx, mask=is_b))
    means_c.append(get_mean_fraction(calo_3d, layer_idx=idx, rawE_idx=rawE_idx, mask=is_c))
    means_tau.append(get_mean_fraction(calo_3d, layer_idx=idx, rawE_idx=rawE_idx, mask=is_tau))


# ------- Plotting -------
labels = list(layers.keys())
x = np.arange(len(labels))  # The label locations: [0, 1, 2, ..., 8]
width = 0.25                # The width of the bars

# Made the figure wider to fit 9 layers comfortably
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the grouped bars
rects1 = ax.bar(x - width, means_b, width, label='b-jets')
rects2 = ax.bar(x, means_c, width, label='c-jets')
rects3 = ax.bar(x + width, means_tau, width, label='tau-jets')

# Formatting
ax.set_ylabel(r'Mean Energy Fraction ($\langle E_{layer} / E_{total} \rangle$)', fontsize=12)
ax.set_xticks(x)

# Rotate labels by 45 degrees for readability
ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')
ax.legend(loc='upper right', frameon=False)

ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
ax.tick_params(which='minor', length=3, direction='in')
ax.tick_params(which='major', length=7, direction='in')

ax.set_yscale('log')

ax.text(0.05, 0.95, atlas_label, 
        transform=ax.transAxes, 
        fontsize=9, 
        verticalalignment='top')

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, max(ymax * 5, 2.0))

# tight_layout ensures the rotated x-labels don't get cut off at the bottom of the image
plt.tight_layout()
plt.draw()
plt.savefig("/home/xzcapfed/MSci/GN3_calo/plots/distributions/full_calorimeter_profile.png", dpi=300)
plt.close()


exit()
# --------- Plotting ----------
global_min = min(np.min(rawE_b), np.min(rawE_c))
global_max = max(np.max(rawE_b), np.max(rawE_c))
if global_min <= 0:
    global_min = 0.000001
logbins = np.geomspace(0.0001, global_max, 30)

savedir="/home/xzcapfed/MSci/GN3_calo/plots/distributions/"
plt.hist(rawE_b/1000, bins=logbins, density=True, histtype='step',color='C0', linewidth=2, linestyle='-')
plt.hist(PresamplerB_b/1000, bins=logbins, density=True, histtype='step',color='C1', linewidth=2, linestyle='-')
plt.hist(EMB1_b/1000, bins=logbins, density=True, histtype='step',color='C2', linewidth=2, linestyle='-')
plt.hist(HEC0_b/1000, bins=logbins, density=True, histtype='step',color='C3', linewidth=2, linestyle='-')
plt.plot([], [], color='C0', label=r'rawE')
plt.plot([], [], color='C1', label=r"PresamplerB")
plt.plot([], [], color='C2', label=r'EMB1')
plt.plot([], [], color='C3', label=r"HEC0")
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
plt.tick_params(which='minor', length=3, direction='in')
plt.tick_params(which='major', length=7, direction='in')
plt.xlabel(r'$p_{T}$ [GeV]', fontsize=12, loc='right')
plt.ylabel('Normalised Number of Jet Objects', fontsize=12, loc='top')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper right', frameon=False)
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax * 1.15)
plt.xlim(xmax=1e5)
plt.text(0.05, 0.95, atlas_label, 
        transform=plt.gca().transAxes, 
        fontsize=9, 
        verticalalignment='top')
plt.figsize=(8, 6)
plt.draw()
plt.savefig(savedir+"rawE", dpi=300)
plt.close()



