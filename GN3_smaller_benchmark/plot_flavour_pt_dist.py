"""
Generates plots a distribution of transverse momentum for each flavour class
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import h5py
import glob
import matplotlib.pyplot as plt

from ftag.hdf5.h5reader import H5Reader
from ftag.cuts import Cuts

from puma.utils import logger
from puma import Histogram, HistogramPlot


#--------- Functions ----------
def load_df(file_pattern, num_jets=None, batch_size=500_000):
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
        ("pt", ">", 20000),
        ("pt", "<", 250000),
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
    ])

    # Only want to plot pT
    jet_vars = [
        "pt", "eta",
        "PartonExtendedTruthLabelID",
        "HadronGhostTruthLabelID",
    ]

    variables = {"jets": jet_vars}

    reader = H5Reader(
        fname=file_paths,
        jets_name="jets",
        batch_size=batch_size,
        do_remove_inf=True,
        shuffle=True
    )

    data = reader.load(variables=variables, cuts=cuts, num_jets=num_jets)

    df = pd.DataFrame(data["jets"])

    return df

def get_details(df, label1, label2):
    '''Defines a boolean array to select the different flavour classes. Returns the arrays as a list'''
    is_light = df[label1] == 0 # light jets = s + ud + g jets
    is_c = df[label1] == 4 # c-jets 
    is_b = df[label1] == 5 # b-jets
    is_s = (df[label1] == 0) & (df[label2] == 3) # strange jets 
    is_ud = (df[label1] ==0) & (df[label2] <= 2) # ud jets
    is_g = (df[label1] ==0) & (df[label2] == 21) # gluon jets
    is_tau = df[label1] == 15 # tau jets 


    return is_light, is_c, is_b, is_s, is_ud, is_g, is_tau

def filter_arr(arr, masks):
    '''Filters an array given a list of boolean mask arrays'''
    
    filtered_arrays = []
    for mask in masks:
        filtered_arr = arr[mask]
        filtered_arrays.append(filtered_arr)

    return filtered_arrays

def plot_hist(test_data, training_data, jet_type, fig_label, save_fname, logx=False, logy=False):
    '''
    Plots a histogram

    Parameters:
    
    test_data : test dataset (numpy array)
    training_data : training dataset (numpy array)
    fig_label : label for the x-axis i.e. leading hadron pT (str)
    save_fname : save name for file (str)
    logx = logarithm plot for the x axis (boolean)
    logy = logarithm plot for the y axis (boolean) 
    '''

    # ensure binning is same for true and contaminated jets
    shared_min = min(test_data.min(), training_data.min())
    shared_max = max(test_data.max(), training_data.max())

    # geometrically sized bins if using log scale on x-axis
    if logx:
        if shared_min == 0:
            shared_min += 1e-10
        bins = np.geomspace(shared_min, shared_max, num=50)
    else:
        bins = np.linspace(shared_min, shared_max, num=50)

    # histogram plotting
    plt.hist(test_data, 
            bins=bins, 
            density=True,
            histtype='step',
            linewidth=2, 
            label=f'Test ${jet_type}$-jet',
            linestyle='--')

    plt.hist(training_data, 
            bins=bins, 
            density=True,
            histtype='step',
            linewidth=2, 
            label=f'Training ${jet_type}$-jet')
    
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

# ----------- Plotting -------------
fname_test='/home/xzcapfed/tmp/ckpts/epoch=003-val_loss=1.01886__test_ttbar.h5' # test data
fname_training='/home/xzcapwsl/phd/datasets/atlas/MSci/GN3V01-6class/output/pp_output_train.h5' # training data

logger.info("Loading h5 files")
df_test= load_df(fname_test)
df_train = load_df(fname_training)

logger.info("Processing datasets")
# define boolean arrays to extract flavours
masks_test = get_details(df_test, "HadronGhostTruthLabelID", "PartonExtendedTruthLabelID")
masks_train = get_details(df_train, "HadronGhostTruthLabelID", "PartonExtendedTruthLabelID")
# extract pT arrays for a given dataset
light_pt_test, c_pt_test, b_pt_test, s_pt_test, ud_pt_test, g_pt_test, tau_pt_test = [pt / 1000 for pt in filter_arr(df_test['pt'], masks_test)] # unpack pT in GeV
light_pt_train, c_pt_train, b_pt_train, s_pt_train, ud_pt_train, g_pt_train, tau_pt_train = [pt / 1000 for pt in filter_arr(df_train['pt'], masks_train)] 

logger.info("Plotting histograms")
# plot distributions using histogram
plot_hist(light_pt_test, light_pt_train, 'light', fig_label='$p_{T}$ [GeV]', save_fname='plots/jet_flavour_distributions/lightjet_pt_ttbar.png', logx=True)
plot_hist(c_pt_test, c_pt_train, 'c', fig_label='$p_{T}$ [GeV]', save_fname='plots/jet_flavour_distributions/cjet_pt_ttbar.png', logx=True)
plot_hist(b_pt_test, b_pt_train, 'b', fig_label='$p_{T}$ [GeV]', save_fname='plots/jet_flavour_distributions/bjet_pt_ttbar.png', logx=True)
plot_hist(s_pt_test, s_pt_train, 's', fig_label='$p_{T}$ [GeV]', save_fname='plots/jet_flavour_distributions/sjet_pt_ttbar.png', logx=True)
plot_hist(ud_pt_test, ud_pt_train, 'ud', fig_label='$p_{T}$ [GeV]', save_fname='plots/jet_flavour_distributions/udjet_pt_ttbar.png', logx=True)
plot_hist(g_pt_test, g_pt_train, 'g', fig_label='$p_{T}$ [GeV]', save_fname='plots/jet_flavour_distributions/gjet_pt_ttbar.png', logx=True)
plot_hist(tau_pt_test, tau_pt_train, '\\tau', fig_label='$p_{T}$ [GeV]', save_fname='plots/jet_flavour_distributions/tau_pt_ttbar.png', logx=True)