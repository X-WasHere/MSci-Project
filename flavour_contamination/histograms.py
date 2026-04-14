"""
This is a script that just plots some simple 1d histogram 
distributions give h5 files 
"""
from __future__ import annotations

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

#--------- Functions ----------
def load_df(file_pattern, variables, dset_name, num_jets=None, batch_size=500_000):
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

    # Full range of cuts to examine whole pT distribution
    # cuts = Cuts.from_list([
    #     ("pt", ">", 20000),
    #     ("pt", "<", 6000000),
    #     ("eta", "<", 2.5),
    #     ("eta", ">", -2.5),
    # ])

    variables = {"calo": variables}

    reader = H5Reader(
        fname=file_paths,
        jets_name=dset_name,
        batch_size=batch_size,
        do_remove_inf=True,
        shuffle=False
    )

    data = reader.load(variables=variables, num_jets=num_jets)

    df = pd.DataFrame(data[dset_name])

    return df

def describe_df(df, column=None):
    """Prints a summary of the given dataframe to the terminal
    
    Parameters:
    df : dataframe table (pd.dataframe)
    column : column to produce statsitics for (str)
    """
    pd.set_option('display.max_columns', None) # show all columns

    print("---SHAPE INFO----")
    print(df.info())
    print("---SEE FIRST FEW ROWS ---")
    print(df.head(5))

    if column:
        print(f"---LOOK at {column}---")
        print(df[column].describe())

    return

def plot_histogram(variable, fig_label, save_fname, logx=False, logy=False, max=None):

    min = variable.min()
    if not max:
        max = variable.max()
    print(f"Data range: {min} to {max}")

    # geometrically sized bins if using log scale on x-axis
    if logx:
        if (max-min) < 0:
            min = 0.0
            max = 1e-5
        if min == 0:
            min += 1e-5
        bins = np.geomspace(min, max, num=50)
    else:
        bins = np.linspace(min, max, num=50)

    plt.hist(variable, 
        bins=bins, 
        density=True, # use density instead of weights for proability density
        # weights=weights_c,
        histtype='step',
        linewidth=2, 
        linestyle='--')

    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')  

    # configuration
    plt.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
    plt.xlabel(fig_label, fontsize=14, loc='right')
    plt.ylabel('Normalized Number of Events', fontsize=14, loc='top')
    plt.legend()
    plt.figsize=(8, 6)
    plt.draw()
    plt.savefig(save_fname)

    plt.close()  

    return 

# ------------- Plotting ----------------
split_label = "HFShowerLabel"
fname='/home/xzcappon/phd/projects/supervising/2025_2026/samples-with-calo-info/user.npond.601589.e8547_s3797_r13144_p7085.tdd.GN3_dev.25_2_76.Haloween2025-27-g1827a5d_output.h5/user.npond.47983433._000003.output.h5'
logger.info("Loading h5 files")
df = load_df(fname, ['EME1'], 'calo')
describe_df(df)


plot_path = "/home/xzcapfed/MSci/GN3_calo/plots"
plot_histogram(df['EME1'], 'test', plot_path+'test')