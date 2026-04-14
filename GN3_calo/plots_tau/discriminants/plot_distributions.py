from __future__ import annotations

import numpy as np
import pandas as pd
import h5py
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

from ftag.hdf5.h5reader import H5Reader
from ftag.cuts import Cuts

from puma.utils import logger
from puma import Histogram, HistogramPlot

#--------- Functions ----------

def get_details(df, label1, label2):
    '''Defines a boolean array to select the different flavour classes'''
    is_light = df[label1] == 0
    is_c = df[label1] == 4
    is_b = df[label1] == 5
    is_tau = df[label1] == 15

    is_ud = (df[label1] ==0) & (df[label2] <= 2)
    is_s = (df[label1] == 0) & (df[label2] == 3)
    is_g = (df[label1] ==0) & (df[label2] == 21)

    return is_light, is_c, is_b, is_ud, is_s, is_g, is_tau

def load_df(file_pattern, num_jets=None, batch_size=500_000, split_label="HFShowerLabel"):
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
    cuts = Cuts.from_list([
        # ("pt", ">", 20000),
        # ("pt", "<", 250000),
        ("pt", ">", 250000),
        ("pt", "<", 6000000),
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
    ])

    # Define required variables
    jet_vars = [
        "pt", "eta",
        "PartonTruthLabelID", "HadronGhostTruthLabelID",
        "GhostBHadronsFinalPt", "GhostBHadronsFinalCount", "GhostCHadronsFinalCount",
        "HadronGhostTruthLabelPt", "HadronGhostTruthLabelLxy", "HadronGhostTruthLabelDR",
        "n_tracks_ghost",
    ]

    variables = {"jets": jet_vars}

    reader = H5Reader(
        fname=file_paths,
        jets_name="jets",
        batch_size=batch_size,
        do_remove_inf=True,
        shuffle=False
    )

    data = reader.load(variables=variables, cuts=cuts, num_jets=num_jets)

    df = pd.DataFrame(data["jets"])

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

def plot_distribution(samples, labels, x_label, atlas_tag, save_fname, logx=False, logy=False):
    '''Plots a histogram for contaminated and true jets for a given variable. For example plots the 
    normalized number of events vs the leading hadron momentum.
    
    Parameters:

    sample : distribution for a selected variable for a jet (Numpy array) 
    jet_type : jet flavour (str)
    fig_label : label for the x-axis i.e. leading hadron pT (str)
    save_fname : save name for file (str)
    logx = logarithm plot for the x axis (boolean)
    logy = logarithm plot for the y axis (boolean) 

    Output:
    
    Produces a histogram plot saved under {save_fname}.
    '''
    min_val = min([sample.min() for sample in samples])
    # max_val = max([sample.max() for sample in samples])
    max_val = 65
    print(f"Data range: {min_val} to {max_val}")

    # geometrically sized bins if using log scale on x-axis
    if logx:
        if min_val == 0:
            min_val += 1e-5
        bins = np.geomspace(min_val, max_val, num=20)
    else:
        bins = np.arange(min_val, max_val)

    # histogram plotting
    for sample, label in zip(samples, labels):
        plt.hist(sample, 
                bins=bins, 
                density=True, # use density instead of weights for proability density
                histtype='step',
                linewidth=2, 
                label=label,
                linestyle='-')
    
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')

    # configuration
    if atlas_tag:
        plt.text(
            0.4, 0.98, # x, y coordinates
            atlas_tag,            
            transform=plt.gca().transAxes,
            fontsize=10, 
            verticalalignment='top', 
            horizontalalignment='left')
    plt.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel(x_label, fontsize=14, loc='right')
    plt.ylabel('Normalized Number of Events', fontsize=14, loc='top')
    plt.legend()
    plt.figsize=(8, 6)
    plt.draw()
    plt.savefig(save_fname)

    plt.close()
    return


#--------- Plotting ----------
fname='/home/xzcapfed/MSci/GN3_calo/logs/new-data-benchmark/ckpts/epoch=004-val_loss=0.97334__test_zprime.h5'
fname_benchmark='/home/xzcapfed/MSci/GN3_calo/logs/benchmark/ckpts/epoch=003-val_loss=0.99337__test_zprime.h5'
# fname='/home/xzcapfed/MSci/GN3_calo/logs/new-data-benchmark/ckpts/epoch=004-val_loss=0.97334__test_ttbar.h5'
# fname_benchmark='/home/xzcapfed/MSci/GN3_calo/logs/benchmark/ckpts/epoch=003-val_loss=0.99337__test_ttbar.h5'

logger.info("Loading h5 files")
df_calo = load_df(fname)
df_benchmark = load_df(fname_benchmark)
logger.info(f"plotting with {len(df_calo)} jets")

is_light, is_c, is_b, is_ud, is_s, is_g, is_tau = get_details(df_calo, "HadronGhostTruthLabelID", "PartonTruthLabelID")
is_light_benchmark, is_c_benchmark, is_b_benchmark, is_ud_benchmark, is_s_benchmark, is_g_benchmark, is_tau_benchmark = get_details(df_benchmark, "HadronGhostTruthLabelID", "PartonTruthLabelID")

if 'ttbar' in fname:
    sample_type = 'ttbar'
    atlas_tag = "$\\sqrt{s}=13$ TeV $t\overline{t}$ events \n20 GeV < $p_T$ < 250 GeV"
else: 
    sample_type = 'zprime'
    atlas_tag = "$\\sqrt{s}=13$ TeV $Z'$ events \n250 GeV < $p_T$ < 6 TeV"

n_tracks = df_calo['n_tracks_ghost']
plot_distribution(samples=[n_tracks[is_c], n_tracks[is_b], n_tracks[is_ud], n_tracks[is_s], n_tracks[is_g], n_tracks[is_tau]],
                  labels=['c-jets', 'b-jets', 'ud-jets', 's-jets', 'gluon-jets', 'tau-jets'],
                  x_label="$n_{tracks}$",
                  atlas_tag=atlas_tag,
                  save_fname=f"/home/xzcapfed/MSci/GN3_calo/plots_tau/discriminants/n_tracks/hist_{sample_type}_calo.png",
                  logx=False
                  )

n_tracks = df_benchmark['n_tracks_ghost']
plot_distribution(samples=[n_tracks[is_c_benchmark], n_tracks[is_b_benchmark], n_tracks[is_ud_benchmark], n_tracks[is_s_benchmark], n_tracks[is_g_benchmark], n_tracks[is_tau_benchmark]],
                  labels=['c-jets', 'b-jets', 'ud-jets', 's-jets', 'gluon-jets', 'tau-jets'],
                  x_label="$n_{tracks}$",
                  atlas_tag=atlas_tag,
                  save_fname=f"/home/xzcapfed/MSci/GN3_calo/plots_tau/discriminants/n_tracks/hist_{sample_type}_benchmark.png",
                  logx=False
                  )