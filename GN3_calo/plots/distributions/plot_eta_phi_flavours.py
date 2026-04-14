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
        "LONGITUDINAL",
        "LATERAL",
        "rawPhi",
        "rawEta"
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
        data['calo']['LONGITUDINAL'],
        data['calo']['LATERAL'],
        data['calo']['rawEta'],
        data['calo']['rawPhi']
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
    flattened_feature = feature_2d[mask].flatten()
    
    return flattened_feature[flattened_feature != 0.0]


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
is_light = df_jets['HadronGhostTruthLabelID'] == 0

# load variables
rawEta_b = filtered_calo_feature(calo_3d, 2, is_b)
rawPhi_b = filtered_calo_feature(calo_3d, 3, is_b)

rawEta_c = filtered_calo_feature(calo_3d, 2, is_c)
rawPhi_c = filtered_calo_feature(calo_3d, 3, is_c)

rawEta_tau = filtered_calo_feature(calo_3d, 2, is_tau)
rawPhi_tau = filtered_calo_feature(calo_3d, 3, is_tau)

rawEta_light = filtered_calo_feature(calo_3d, 2, is_light)
rawPhi_light = filtered_calo_feature(calo_3d, 3, is_light)

 
# histograms 
atlas_label = (
    r"$\mathbfit{ATLAS}$ Simulation Preliminary" + "\n"
    r"$\sqrt{s} = 13.6$ TeV" + "\n"
    r" $t\bar{t}$ events, 20 GeV < $p_{T}$ < 250 GeV" + "\n"
    r" $Z'$ events, 250 GeV < $p_{T}$ < 6 TeV"
)

savedir="/home/xzcapfed/MSci/GN3_calo/plots/distributions/"

plt.hist(rawEta_b, bins=30, density=True, histtype='step',color='C0', linewidth=2, linestyle='-')
plt.hist(rawEta_tau, bins=30, density=True, histtype='step',color='C1', linewidth=2, linestyle='-')
plt.hist(rawEta_light, bins=30, density=True, histtype='step',color='C2', linewidth=2, linestyle='-')
plt.plot([], [], color='C0', label=r'$b$-jets')
plt.plot([], [], color='C2', label=r'$\tau$-jets')
plt.plot([], [], color='C3', label=r"Light-jets")
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
plt.tick_params(which='minor', length=3, direction='in')
plt.tick_params(which='major', length=7, direction='in')
plt.xlabel(r'raw $\eta$', fontsize=12, loc='right')
plt.ylabel('Normalised Number of Jet Objects', fontsize=12, loc='top')
plt.legend(loc='upper right', frameon=False)
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax * 1.15)
plt.text(0.05, 0.95, atlas_label, 
        transform=plt.gca().transAxes, 
        fontsize=9, 
        verticalalignment='top')
plt.figsize=(8, 6)
plt.draw()
plt.savefig(savedir+"rawEta_by_flavour", dpi=300)
plt.close()

plt.hist(rawPhi_b, bins=30, density=True, histtype='step',color='C0', linewidth=2, linestyle='-')
plt.hist(rawPhi_tau, bins=30, density=True, histtype='step',color='C1', linewidth=2, linestyle='-')
plt.hist(rawPhi_light, bins=30, density=True, histtype='step',color='C2', linewidth=2, linestyle='-')
plt.plot([], [], color='C0', label=r'$b$-jets')
plt.plot([], [], color='C2', label=r'$\tau$-jets')
plt.plot([], [], color='C3', label=r"Light-jets")
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
plt.tick_params(which='minor', length=3, direction='in')
plt.tick_params(which='major', length=7, direction='in')
plt.xlabel(r'raw $\phi$ [rad]', fontsize=12, loc='right')
plt.ylabel('Normalised Number of Jet Objects', fontsize=12, loc='top')
plt.legend(loc='upper right', frameon=False)
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax * 1.3)
plt.text(0.05, 0.95, atlas_label, 
        transform=plt.gca().transAxes, 
        fontsize=9, 
        verticalalignment='top')

plt.figsize=(8, 6)
plt.draw()
plt.savefig(savedir+"rawPhi_by_flavour.png", dpi=300)
plt.close()