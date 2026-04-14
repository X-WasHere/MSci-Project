"""
Examine optimal cut parameters.

Note: for the JZ sample dataset HFShowerLabel -> HFGluonSpiltLabel
"""
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

#--------- Functions ----------
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
        ("pt", ">", 20000),
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
        split_label
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

def extract_variable(df, is_c, is_b, variable, split_label="HFShowerLabel"):
    '''Returns true and contaminated jet arrays for b- and c- jets and particular varible of interest
    for histogram plotting'''

    # Extract contaminated and true b/c jet arrays
    contaminated_bjet = df[split_label][is_b] == 1 # is_b is a mask for b-hadrons
    contaminated_cjet = df[split_label][is_c] == 1 
    true_bjet = df[split_label][is_b] == 0 
    true_cjet = df[split_label][is_c] == 0 
    
    b_variable = df[variable][is_b] # extract given variable for b jets (i.e b-jet momentum)
    contaminated_bjet_variable = b_variable[contaminated_bjet]
    true_bjet_variable = b_variable[true_bjet]

    c_variable = df[variable][is_c] # extract given variable for c jets (i.e c-jet momentum)
    contaminated_cjet_variable = c_variable[contaminated_cjet]
    true_cjet_variable = c_variable[true_cjet]

    return contaminated_bjet_variable, true_bjet_variable, contaminated_cjet_variable, true_cjet_variable

def plot_2d_distribution(contam_var1, contam_var2, true_var1, true_var2, 
                         jet_type, x_label, y_label, save_fname, 
                         logx=False, logy=False, logz=True, cut=None):
    '''
    Plots side-by-side 2D histograms (heatmaps) for Contaminated vs True jets.
    
    Parameters:
    contam_var1, contam_var2 : Arrays for X and Y variables (Contaminated Jets)
    true_var1, true_var2     : Arrays for X and Y variables (True Jets)
    jet_type : str (e.g., 'b')
    x_label, y_label : Labels for the axes
    save_fname : filename
    logx, logy : Log scale for X and Y axes
    logz : Log scale for the COLOR (density). Highly recommended for jet data.
    '''

    # 1. Clean Data (Remove NaNs/Infs from PAIRS)
    # We must ensure that if index i is bad in var1, it is removed from var2 as well.
    mask_c = np.isfinite(contam_var1) & np.isfinite(contam_var2)
    c1, c2 = contam_var1[mask_c], contam_var2[mask_c]

    mask_t = np.isfinite(true_var1) & np.isfinite(true_var2)
    t1, t2 = true_var1[mask_t], true_var2[mask_t]

    if len(c1) == 0 or len(t1) == 0:
        print(f"Skipping {save_fname}: Data empty after filtering NaNs.")
        return

    # shared bins for x-axis
    global_min_x = min(c1.min(), t1.min())
    # global_max_x = max(c1.max(), t1.max())
    global_max_x = 3.0

    if logx:
        # Handle 0 or negative values for log scale
        pos_x = np.concatenate([c1, t1])
        pos_x = pos_x[pos_x > 0]
        min_x = pos_x.min() if len(pos_x) > 0 else 0.1
        bins_x = np.geomspace(min_x, global_max_x, num=100)
    else:
        bins_x = np.linspace(global_min_x, global_max_x, num=100)

    global_min_y = min(c2.min(), t2.min())
    global_max_y = max(c2.max(), t2.max())

    if logy:
        pos_y = np.concatenate([c2, t2])
        pos_y = pos_y[pos_y > 0]
        min_y = pos_y.min() if len(pos_y) > 0 else 0.1
        bins_y = np.geomspace(min_y, global_max_y, num=100)
    else:
        bins_y = np.linspace(global_min_y, global_max_y, num=100)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    plot_args = {
        'bins': [bins_x, bins_y],
        'density': True,          # Normalize so integral = 1
        'cmap': 'viridis',
        'norm': LogNorm() if logz else None
    }

    # contaminated jets
    h1 = axes[0].hist2d(c1, c2, **plot_args)
    axes[0].set_title(f'Contaminated ${jet_type}$-jet ({len(c1)} jets)')
    # true jets 
    h2 = axes[1].hist2d(t1, t2, **plot_args)
    axes[1].set_title(f'True ${jet_type}$-jet ({len(t1)} jets)')

    # Formatting
    for ax in axes:
        ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
        ax.set_xlabel(x_label, fontsize=14)
        if logx: ax.set_xscale('log')
        if logy: ax.set_yscale('log')

    axes[0].set_ylabel(y_label, fontsize=14)

    # grab the image from the second plot (h2[3]) to generate the bar
    cbar = fig.colorbar(h2[3], ax=axes.ravel().tolist())
    cbar.set_label('Probability Density', fontsize=14)

    if cut: 
        _, __, xline, yline = define_cut(global_min_x, cut[0], cut[1], global_min_y, logx)
        axes[0].plot(xline, yline, color='red', linestyle='--')
        _, __, xline, yline = define_cut(global_min_x, cut[0], cut[1], global_min_y, logx)
        axes[1].plot(xline, yline, color='red', linestyle='--')

    plt.savefig(save_fname, bbox_inches='tight')
    plt.close()
    return

def plot_2d_ratio_evaluation(contaminated_x, contaminated_y, true_x, true_y, jet_type, x_label, y_label, save_fname,
                             extra_samples, cut=None, logx=False, logy=False, logz=False):
    '''
    Plots a single probability density distribuition given two variables for two samples 
    to calulate a ratio. Also evaluates based true bjet efficiency and contaminated fraction 
    based on given cut parameters.
    
    Parameters:
    contaminated_x, contaminated_y : Arrays for X and Y variables (Contaminated Jets)
    true_x, true_y     : Arrays for X and Y variables (True Jets)
    jet_type : str (e.g., 'b')
    x_label, y_label : Labels for the axes
    save_fname : filename
    logx, logy : Log scale for X and Y axes
    logz : Log scale for the probability density
    cut : array containing two values to define a cut on x-axis and y-axis 
    '''
    # clean dataset
    mask_c = np.isfinite(contaminated_x) & np.isfinite(contaminated_y)
    cx, cy = contaminated_x[mask_c], contaminated_y[mask_c]
    mask_t = np.isfinite(true_x) & np.isfinite(true_y)
    tx, ty = true_x[mask_t], true_y[mask_t]
    if len(cx) == 0 or len(tx) == 0:
        print(f"Skipping {save_fname}: Data empty after filtering NaNs.")
        return
    
    global_min_x = min(cx.min(), tx.min())
    print(f"\nGLOBAL MIN X: {global_min_x}")
    global_max_x = max(cx.max(), tx.max())
    # global_max_x = 7.5
    if logx:
        pos_x = np.concatenate([cx, tx])
        pos_x = pos_x[pos_x > 0] # filter +ve values for log
        min_x = pos_x.min() if len(pos_x) > 0 else 0.1
        bins_x = np.geomspace(min_x, global_max_x, num=100)
    else:
        bins_x = np.linspace(global_min_x, global_max_x, num=100)

    global_min_y = min(cy.min(), ty.min())
    print(f"GLOBAL MIN Y: {global_min_y}")
    global_max_y = max(cy.max(), ty.max())
    if logy:
        pos_y = np.concatenate([cy, ty])
        pos_y = pos_y[pos_y > 0]
        min_y = pos_y.min() if len(pos_y) > 0 else 0.1
        bins_y = np.geomspace(min_y, global_max_y, num=100)
    else:
        bins_y = np.linspace(global_min_y, global_max_y, num=100)

    hist_contaminated, xedges, yedges = np.histogram2d(cx, cy, bins=[bins_x, bins_y], density=True)
    hist_true, _, _ = np.histogram2d(tx, ty, bins=[bins_x, bins_y], density=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        hist_ratio = np.divide(hist_contaminated, hist_true)
        # mask NaNs (0/0) and Infs (N/0)
        hist_ratio = np.ma.masked_invalid(hist_ratio) 
        # mask zeros (0/N) so that LogNorm doesn't error
        if logz:
            hist_ratio = np.ma.masked_less_equal(hist_ratio, 0)
        
    if hist_ratio.count() == 0:
        print(f"SKIPPING PLOT: No overlap between datasets in {save_fname}")
        print("This means there is no bin where BOTH contaminated and true jets exist.")
        return

    fig, ax = plt.subplots(figsize=(8,6))
    
    # plot precomupted histogram
    mesh = ax.pcolormesh(xedges, yedges, hist_ratio.T, cmap='viridis', norm=LogNorm() if logz else None)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Probability Density")

    if cut: 
        m, c, xline, yline = define_cut(xedges[0], cut[0], cut[1], yedges[0], logx)
        print(f"\nm : {m} and c: {c}")
        ax.plot(xline, yline, color='red', linestyle='--')
        cut_contaminated = is_contaminated(true_x, true_y, m, c)
        true_bjet_efficiency = (len(true_x) - len(true_x[cut_contaminated])) / len(true_x)

        # contaminated jet processing
        cut_contaminated_JZ = is_contaminated(contaminated_x, contaminated_y, m, c)
        cut_true_JZ = is_contaminated(extra_samples[0], extra_samples[1], m, c)
        contaminated_fraction = (len(contaminated_x[cut_contaminated_JZ])) / (len(contaminated_x[cut_contaminated_JZ]) + len(extra_samples[0][cut_true_JZ]))

        contaminated_JZ_efficiency = (len(contaminated_x[cut_contaminated_JZ]) + len(extra_samples[0][cut_true_JZ])) / (len(contaminated_x) + len(extra_samples[0]))

    # plt.title(f"Ratio plot of Contaminted {jet_type}-jets : True {jet_type}-jets")
    # plt.savefig(save_fname, bbox_inches='tight')
    # plt.close()
    
    return true_bjet_efficiency, contaminated_JZ_efficiency, contaminated_fraction

def define_cut(x1, x2, y1, y2, logx):
    '''
    Determines the gradient and y-intercept of a cut given two x values and 
    two y values. Also returns an x array and y array according to the linear 
    equation which can be used for plotting.
    '''
    if logx:
        m = (y2-y1)/(np.log10(x2) - np.log10(x1))
        c = y1 - m*np.log10(x1)

        x_line = np.logspace(np.log10(x1), np.log10(x2), 100)
        y_line = m*np.log10(x_line) + c
    else:
        m = (y2-y1)/(x2 - x1)
        c = y1 - m*x1

        x_line = np.linspace(x1, x2, 100)
        y_line = m*x_line + c

    return m, c, x_line, y_line

def is_contaminated(x, y, m, c):
    '''
    Given an input arrays of x and y variables and parameters defining the equation y = m log(x) + c
    returns a boolean array which is True if values are below the equation and False otherwise
    '''
    y_eq = m * np.log(x) + c 
    contamination_label = y < y_eq

    return contamination_label 

#------------Plotting-------------
split_label_JZ = "HFGluonSplitLabel"
split_label_ttbar = "HFShowerLabel"
fname_JZ='/home/xzcapfed/MSci/flavour_contamination/sample_datasets/JZ_output_801171.h5'
fname_ttbar='/home/xzcapfed/MSci/flavour_contamination/sample_datasets/ttbar_test_mc20.h5'
logger.info("Loading h5 files")
df_JZ = load_df(fname_JZ, split_label=split_label_JZ)
df_ttbar = load_df(fname_ttbar, split_label=split_label_ttbar)
describe_df(df_JZ, column='pt')

# Extract b-jets and c-jets with boolean array
is_c_JZ = df_JZ["HadronGhostTruthLabelID"] == 4
is_b_JZ = df_JZ["HadronGhostTruthLabelID"] == 5

is_c_ttbar = df_ttbar["HadronGhostTruthLabelID"] == 4
is_b_ttbar = df_ttbar["HadronGhostTruthLabelID"] == 5

logger.info("Extracting distributions")
# jet pT
contaminated_bjet_pT_JZ, true_bjet_pT_JZ, _, __ = extract_variable(df_JZ, is_c_JZ, is_b_JZ, "pt", split_label_JZ)
contaminated_bjet_pT_ttbar, true_bjet_pT_ttbar, _, __ = extract_variable(df_ttbar, is_c_ttbar, is_b_ttbar, "pt", split_label_ttbar)
# b-jet pT
contaminated_bjet_ghostpt_JZ, true_bjet_ghostpt_JZ, _, __ = extract_variable(df_JZ, is_c_JZ, is_b_JZ, 'GhostBHadronsFinalPt', split_label_JZ) 
contaminated_bjet_ghostpt_ttbar, true_bjet_ghostpt_ttbar, _, __ = extract_variable(df_ttbar, is_c_ttbar, is_b_ttbar, 'GhostBHadronsFinalPt', split_label_ttbar) 
# dR
contaminated_bjet_leadingdr_JZ, true_bjet_leadingdr_JZ, _, __ = extract_variable(df_JZ, is_c_JZ, is_b_JZ, 'HadronGhostTruthLabelDR', split_label_JZ)
contaminated_bjet_leadingdr_ttbar, true_bjet_leadingdr_ttbar, _, __ = extract_variable(df_ttbar, is_c_ttbar, is_b_ttbar, 'HadronGhostTruthLabelDR', split_label_ttbar)
# b-hadrdon pT / jet pT
contaminated_bpt_jetpt_JZ = contaminated_bjet_ghostpt_JZ / contaminated_bjet_pT_JZ
true_bpt_jetpt_JZ = true_bjet_ghostpt_JZ / true_bjet_pT_JZ
contaminated_bpt_jetpt_ttbar = contaminated_bjet_ghostpt_ttbar / contaminated_bjet_pT_ttbar
true_bpt_jetpt_ttbar = true_bjet_ghostpt_ttbar / true_bjet_pT_ttbar


logger.info("Plotting")
plot_path='plots_combined/JZ801171_contaminated_ttbar_true/'
xcut = 0.115385
ycut = 0.692308
true_bjet_efficiency, contaminated_JZ_efficiency, contaminated_fraction = plot_2d_ratio_evaluation(
                                            contaminated_bpt_jetpt_JZ, 
                                            contaminated_bjet_leadingdr_JZ, 
                                            np.concatenate((contaminated_bpt_jetpt_JZ, true_bpt_jetpt_ttbar)),
                                            np.concatenate((contaminated_bjet_leadingdr_JZ, true_bjet_leadingdr_ttbar)),
                                            jet_type='b',
                                            x_label='$b$-hardron $p_{T}$ / jet $p_{T}$',
                                            y_label='Leading hadron $\Delta R$',
                                            save_fname=plot_path+'TEST.png',
                                            extra_samples=[true_bpt_jetpt_JZ, true_bjet_leadingdr_JZ],
                                            cut = (xcut, ycut),
                                            logx=True,
                                            logy=False,
                                            logz=True)


x_cuts = np.linspace(0.1, 0.3, 14)
y_cuts = np.linspace(0.6, 0.8, 14)

# rows = []
# for i in range(len(x_cuts)):
#     for j in range(len(y_cuts)):
#         if (i+j) % 10 == 0:
#             print(f"Step {(i*len(y_cuts)) + (j+1)} / {len(x_cuts)*len(y_cuts)}")
#         true_bjet_efficiency, contaminated_JZ_efficiency, contaminated_fraction = plot_2d_ratio_evaluation(
#                                                     contaminated_bpt_jetpt_JZ, 
#                                                     contaminated_bjet_leadingdr_JZ, 
#                                                     np.concatenate((contaminated_bpt_jetpt_JZ, true_bpt_jetpt_ttbar)),
#                                                     np.concatenate((contaminated_bjet_leadingdr_JZ, true_bjet_leadingdr_ttbar)),
#                                                     jet_type='b',
#                                                     x_label='$b$-hardron $p_{T}$ / jet $p_{T}$',
#                                                     y_label='Leading hadron $\Delta R$',
#                                                     save_fname=plot_path+'TEST.png',
#                                                     extra_samples=[true_bpt_jetpt_JZ, true_bjet_leadingdr_JZ],
#                                                     cut = (x_cuts[i], y_cuts[j]),
#                                                     logx=True,
#                                                     logy=False,
#                                                     logz=True)
#         row_data = {
#             'true_bjet_efficiency' : true_bjet_efficiency,
#             'contaminated_JZ_efficiency' : contaminated_JZ_efficiency,
#             'contaminated_fraction' : contaminated_fraction,
#             'x_cut' : x_cuts[i],
#             'y_cut' : y_cuts[j]
#             }
#         rows.append(row_data)

# efficiencies = pd.DataFrame(rows)
# save_file = os.path.join(plot_path, 'total_efficiencies.pkl')
# efficiencies.to_pickle(save_file)

# x_cuts = np.linspace(0.1, 0.3, 22)
# y_cuts = np.linspace(0.6, 0.8, 22)
# true_bjet_efficiencies = []
# contaminated_fractions = []
# for i in range(len(x_cuts)):
#     if i % 10 == 0:
#         print(f"Step {i} / {len(x_cuts)}")
#     true_bjet_efficiency, contaminated_JZ_efficiency, contaminated_fraction = plot_2d_ratio_evaluation(
#                                                     contaminated_bpt_jetpt_JZ, 
#                                                     contaminated_bjet_leadingdr_JZ, 
#                                                     np.concatenate((contaminated_bpt_jetpt_JZ, true_bpt_jetpt_ttbar)),
#                                                     np.concatenate((contaminated_bjet_leadingdr_JZ, true_bjet_leadingdr_ttbar)),
#                                                     jet_type='b',
#                                                     x_label='$b$-hardron $p_{T}$ / jet $p_{T}$',
#                                                     y_label='Leading hadron $\Delta R$',
#                                                     save_fname=plot_path+'TEST.png',
#                                                     extra_samples=[true_bpt_jetpt_JZ, true_bjet_leadingdr_JZ],
#                                                     cut = (x_cuts[i], y_cuts[i]),
#                                                     logx=True,
#                                                     logy=False,
#                                                     logz=True)
#     true_bjet_efficiencies.append(true_bjet_efficiency)
#     contaminated_fractions.append(contaminated_fraction)

# xvals = x_cuts + y_cuts
# plt.figure(figsize=(8,6))
# plt.plot(xvals, true_bjet_efficiencies, '.', label='true bjet efficiency')
# plt.plot(xvals, contaminated_fractions, '.', label='contaminated fraction')
# plt.ylabel("Fraction")
# plt.xlabel("Cut depth")
# plt.legend(loc='best')
# plt.title("Cut metrics as a function of cut depth")
# plt.savefig(plot_path+'efficiencies_TEST.png')

