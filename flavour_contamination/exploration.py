"""
This is an initial script exploring the dataset for flavour contaminated jets.
The outputs are histograms showing distributions, 2D histograms are also produced.

Looks at the new HFShowerLabel: 0 for all light and tau jets, for b and c jets it can be 0 or 1. 
- If this label is 0, then it means we think this is a normal b-jet. 
- If its 1, it means I think its something caused by a gluon split rather than coming from the main physics process.

Note: for the JZ sample dataset HFShowerLabel -> HFGluonSpiltLabel
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

def plot_distribution(contaminated_jet, true_jet, jet_type, fig_label, save_fname, logx=False, logy=False, shared_max=None):
    '''Plots a histogram for contaminated and true jets for a given variable. For example plots the 
    normalized number of events vs the leading hadron momentum.
    
    Parameters:

    contaminated_jets : distribution for a selected variable for a jet labelled as contaminted (Numpy array) 
    true_jets : distribution for a selected variable for a jet labelled as true (Numpy array) 
    jet_type : jet flavour (str)
    fig_label : label for the x-axis i.e. leading hadron pT (str)
    save_fname : save name for file (str)
    logx = logarithm plot for the x axis (boolean)
    logy = logarithm plot for the y axis (boolean) 

    Output:
    
    Produces a histogram plot saved under {save_fname}.
    '''
    print("\n", fig_label)
    # ensure binning is same for true and contaminated jets
    shared_min = min(contaminated_jet.min(), true_jet.min())
    if not shared_max:
        shared_max = max(contaminated_jet.max(), true_jet.max())
    print(f"Data range: {shared_min} to {shared_max}")

    # geometrically sized bins if using log scale on x-axis
    if logx:
        if (shared_max-shared_min) < 0:
            shared_min = 0.0
            shared_max = 1e-5
        if shared_min == 0:
            shared_min += 1e-5
        bins = np.geomspace(shared_min, shared_max, num=50)
    else:
        bins = np.linspace(shared_min, shared_max, num=50)

    # weights_c = np.ones_like(contaminated_jet) / len(contaminated_jet)
    # weights_t = np.ones_like(true_jet) / len(true_jet)

    # histogram plotting
    plt.hist(contaminated_jet, 
            bins=bins, 
            density=True, # use density instead of weights for proability density
            # weights=weights_c,
            histtype='step',
            linewidth=2, 
            label=f'Light-flavour contaminted ${jet_type}$-jet',
            linestyle='--')

    plt.hist(true_jet, 
            bins=bins, 
            density=True,
            # weights=weights_t,
            histtype='step',
            linewidth=2, 
            label=f'True ${jet_type}$-jet')
    
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

def plot_2d_distribution(contam_var1, contam_var2, true_var1, true_var2, 
                         jet_type, x_label, y_label, save_fname, 
                         atlas_tag1=None, atlas_tag2=None,
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
    logz : Log scale for the density
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
    
    # weights_c = np.ones_like(c1) / len(c1)
    # weights_t = np.ones_like(t1) / len(t1)

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

    # ATLAS tags
    if atlas_tag1:
        axes[0].text(0.05, 0.98, atlas_tag1, transform=axes[0].transAxes, 
                     fontsize=9, verticalalignment='top')
    if atlas_tag2:
        axes[1].text(0.05, 0.98, atlas_tag2, transform=axes[1].transAxes, 
                     fontsize=9, verticalalignment='top')

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

    plt.savefig(save_fname, bbox_inches='tight', dpi=300)
    plt.close()
    return

def plot_2d_ratio_distribution(contaminated_x, contaminated_y, true_x, true_y, jet_type, x_label, y_label, save_fname,
                               atlas_tag=None, logx=False, logy=False, logz=False, cut=None):
    '''
    Plots side-by-side 2D histograms (heatmaps) for Contaminated vs True jets.
    
    Parameters:
    contaminated_x, contaminated_y : Arrays for X and Y variables (Contaminated Jets)
    true_x, true_y     : Arrays for X and Y variables (True Jets)
    jet_type : str (e.g., 'b')
    x_label, y_label : Labels for the axes
    save_fname : filename
    logx, logy : Log scale for X and Y axes
    logz : Log scale for the COLOR (density)
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
        _, __, xline, yline = define_cut(xedges[0], cut[0], cut[1], yedges[0], logx)
        ax.plot(xline, yline, color='red', linestyle='--')

    if atlas_tag:
        ax.text(0.05, 0.98, atlas_tag, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top')

    plt.title(f"Ratio plot of Contaminted {jet_type}-jets : True {jet_type}-jets")
    plt.savefig(save_fname, bbox_inches='tight', dpi=300)
    plt.close()
    
    return 

def define_cut(x1, x2, y1, y2, logx):
    if logx:
        m = (y2-y1)/(np.log10(x2) - np.log10(x1))
        c = y1 - m*np.log10(x1)

        x_line = np.logspace(np.log10(x1), np.log10(x2), 100)
        y_line = m*np.log10(x_line) + c
    else:
        m = (y2-y1)/(np.log10(x2) - np.log10(x1))
        c = y1 - m*np.log10(x1)

        x_line = np.logspace(np.log10(x1), np.log10(x2), 100)
        y_line = m*np.log10(x_line) + c

    return m, c, x_line, y_line
#------------Plotting-------------
split_label = "HFGluonSplitLabel"
# split_label = "HFShowerLabel"
fname='/home/xzcapfed/MSci/flavour_contamination/sample_datasets/JZ_output_801171.h5'
logger.info("Loading h5 files")
df = load_df(fname, split_label=split_label)
describe_df(df, column='pt')

# Extract b-jets and c-jets with boolean array
is_c = df["HadronGhostTruthLabelID"] == 4
is_b = df["HadronGhostTruthLabelID"] == 5

logger.info("Extracting distributions")
# extract distributions of variables 
contaminated_bjet_pT, true_bjet_pT, contaminated_cjet_pT, true_cjet_pT = extract_variable(df, is_c, is_b, "pt", split_label)
contaminated_bjet_eta, true_bjet_eta, contaminated_cjet_eta, true_cjet_eta = extract_variable(df, is_c, is_b, 'eta', split_label)
contaminated_bjet_ghostpt, true_bjet_ghostpt, _, __ = extract_variable(df, is_c, is_b, 'GhostBHadronsFinalPt', split_label) 
contaminated_bjet_ghostcount, true_bjet_ghostcount, _, __ = extract_variable(df, is_c, is_b, 'GhostBHadronsFinalCount', split_label) # bit of a work around
_, __, contaminated_cjet_ghostcount, true_cjet_ghostcount = extract_variable(df, is_c, is_b, 'GhostCHadronsFinalCount', split_label)
# find the leading hadron in this jet, and then get its pT,Lxy,dR
contaminated_bjet_leadingpt, contaminated_bjet_leadingpt, contaminated_cjet_leadingpt, true_cjet_leadingpt = extract_variable(df, is_c, is_b, 'HadronGhostTruthLabelPt', split_label)
contaminated_bjet_leadinglxy, true_bjet_leadinglxy, contaminated_cjet_leadinglxy, true_cjet_leadinglxy = extract_variable(df, is_c, is_b, 'HadronGhostTruthLabelLxy', split_label) # transverse decay length
contaminated_bjet_leadingdr, true_bjet_leadingdr, contaminated_cjet_leadingdr, true_cjet_leadingdr = extract_variable(df, is_c, is_b, 'HadronGhostTruthLabelDR', split_label) # delta R
# b-hadrdon pT / jet pT
contaminated_bpt_jetpt = contaminated_bjet_ghostpt / contaminated_bjet_pT
true_bpt_jetpt = true_bjet_ghostpt / true_bjet_pT

plot_path = "plots_zprime_mc20/"

# single variable histograms  
logger.info("Plotting")
# plot_distribution(contaminated_bjet_pT/1000, true_bjet_pT/1000, 'b', fig_label='$p_{T}$ [GeV]', save_fname=plot_path+'pt_dist_bjet.png')
# plot_distribution(contaminated_cjet_pT/1000, true_cjet_pT/1000, 'c', fig_label='$p_{T}$ [GeV]', save_fname=plot_path+'pt_dist_cjet.png')

# plot_distribution(contaminated_bjet_eta, true_bjet_eta, 'b', fig_label='$\eta$', save_fname=plot_path+'eta_dist_bjet.png')
# plot_distribution(contaminated_cjet_eta, true_cjet_eta, 'c', fig_label='$\eta$', save_fname=plot_path+'eta_dist_cjet.png')

# plot_distribution(contaminated_bjet_ghostpt/1000, true_bjet_ghostpt/1000, 'b', fig_label='$p_{T}$ [GeV]', save_fname=plot_path+'ghostpt_dist_bjet.png', logx=True)
# # plot_distribution(contaminated_cjet_ghostpt/1000, true_cjet_ghostpt/1000, 'c', fig_label='$p_{T}$ [GeV]', save_fname=plot_path+'ghostpt_dist_cjet.png', logx=True)

# plot_distribution(contaminated_bjet_ghostcount, true_bjet_ghostcount, 'b', fig_label='Ghost $b$-hadron final count', save_fname=plot_path+'ghostcount_dist_bjet.png', logy=True)
# plot_distribution(contaminated_cjet_ghostcount, true_cjet_ghostcount, 'c', fig_label='Ghost $c$-hadron final count', save_fname=plot_path+'ghostcount_dist_cjet.png', logy=True)

# plot_distribution(contaminated_bjet_leadingpt/1000, true_bjet_leadingpt/1000, 'b', fig_label='Leading hadron $p_{T}$ [GeV]', save_fname=plot_path+'leadingpt_dist_bjet.png', logx=True)
# plot_distribution(contaminated_cjet_leadingpt/1000, true_cjet_leadingpt/1000, 'c', fig_label='Leading hadron $p_{T}$ [GeV]', save_fname=plot_path+'leadingpt_dist_cjet.png', logx=True)

# plot_distribution(contaminated_bjet_leadinglxy, true_bjet_leadinglxy, 'b', fig_label='Leading hadron $L_{xy}$ [mm]', save_fname=plot_path+'leadinglxy_dist_bjet.png', logx=True)
# plot_distribution(contaminated_cjet_leadinglxy, true_cjet_leadinglxy, 'c', fig_label='Leading hadron $L_{xy}$ [mm]', save_fname=plot_path+'leadinglxy_dist_cjet.png', logx=True)

# plot_distribution(contaminated_bjet_leadingdr, true_bjet_leadingdr, 'b', fig_label='Leading hadron $\Delta R$', save_fname=plot_path+'leadingdr_dist_bjet.png')
# plot_distribution(contaminated_cjet_leadingdr, true_cjet_leadingdr, 'c', fig_label='Leading hadron $\Delta R$', save_fname=plot_path+'leadingdr_dist_cjet.png')

# # composite variable
# plot_distribution(contaminted_bpt_jetpt, true_bpt_jetpt, 'b', fig_label='$b$-hardron $p_{T}$ / jet $p_{T}$', save_fname=plot_path+'bpt_to_jetpt.png', shared_max=3.0)

# 2D histograms
# plot_2d_distribution(contaminated_bjet_ghostpt / 1000, 
#                      contaminated_bjet_leadingdr, 
#                      true_bjet_ghostpt / 1000,
#                      true_bjet_leadingdr,
#                      jet_type='b',
#                      x_label='$b$-hardron $p_{T}$ [GeV]',
#                      y_label='Leading hadron $\Delta R$',
#                      save_fname=plot_path+'2d_bpt_dr.png',
#                      logx=True,
#                      logy=False,
#                      logz=True)

# plot_2d_distribution(contaminated_bjet_ghostpt / 1000, 
#                      contaminated_bjet_pT / 1000, 
#                      true_bjet_ghostpt / 1000,
#                      true_bjet_pT / 1000,
#                      jet_type='b',
#                      x_label='$b$-hardron $p_{T}$ [GeV]',
#                      y_label='Jet $p_{T}$ [GeV]',
#                      save_fname=plot_path+'2d_bpt_jetpt.png',
#                      logx=True,
#                      logy=True,
#                      logz=True)

# plot_2d_distribution(contaminated_bjet_pT / 1000, 
#                      contaminated_bjet_leadingdr, 
#                      true_bjet_pT / 1000,
#                      true_bjet_leadingdr,
#                      jet_type='b',
#                      x_label='Jet $p_{T}$ [GeV]',
#                      y_label='Leading hadron $\Delta R$',
#                      save_fname=plot_path+'2d_jetpt_dr.png',
#                      logx=True,
#                      logy=False,
#                      logz=True)

# plot_2d_distribution(contaminted_bpt_jetpt / 1000, 
#                      contaminated_bjet_leadingdr, 
#                      true_bpt_jetpt / 1000,
#                      true_bjet_leadingdr,
#                      jet_type='b',
#                      x_label='$b$-hardron $p_{T}$ / jet $p_{T}$',
#                      y_label='Leading hadron $\Delta R$',
#                      save_fname=plot_path+'2d_bpt_jetpt_dr.png',
#                      logx=True,
#                      logy=False,
#                      logz=True)

# Ratio 2d histograms 
plot_path = "plots_JZ/plots_JZ_801172/ratio_histograms/"
# plot_2d_ratio_distribution(contaminated_bjet_ghostpt / 1000, 
#                            contaminated_bjet_leadingdr, 
#                            true_bjet_ghostpt / 1000,
#                            true_bjet_leadingdr,
#                            jet_type='b',
#                            x_label='$b$-hardron $p_{T}$ [GeV]',
#                            y_label='Leading hadron $\Delta R$',
#                            save_fname=plot_path+'2d_bpt_dr.png',
#                            logx=True,
#                            logy=False,
#                            logz=False)

# plot_2d_ratio_distribution(contaminated_bjet_ghostpt / 1000, 
#                            contaminated_bjet_pT / 1000, 
#                            true_bjet_ghostpt / 1000,
#                            true_bjet_pT,
#                            jet_type='b',
#                            x_label='$b$-hardron $p_{T}$ [GeV]',
#                            y_label='Jet $p_{T}$ [GeV]',
#                            save_fname=plot_path+'2d_bpt_jetpt.png',
#                            logx=True,
#                            logy=True,
#                            logz=False)

# plot_2d_ratio_distribution(contaminated_bjet_pT / 1000, 
#                            contaminated_bjet_leadingdr, 
#                            true_bjet_pT / 1000,
#                            true_bjet_leadingdr,
#                            jet_type='b',
#                            x_label='Jet $p_{T}$ [GeV]',
#                            y_label='Leading hadron $\Delta R$',
#                            save_fname=plot_path+'2d_jetpt_dr.png',
#                            logx=True,
#                            logy=False,
#                            logz=True)

# plot_2d_ratio_distribution(contaminted_bpt_jetpt / 1000, 
#                      contaminated_bjet_leadingdr, 
#                      true_bpt_jetpt / 1000,
#                      true_bjet_leadingdr,
#                      jet_type='b',
#                      x_label='$b$-hardron $p_{T}$ / jet $p_{T}$',
#                      y_label='Leading hadron $\Delta R$',
#                      save_fname=plot_path+'2d_bpt_jetpt_dr.png',
#                      logx=True,
#                      logy=False,
#                      logz=False)

# ---- Ratio plot for combined ttbar and JZ -----
split_label = "HFShowerLabel"
fname='/home/xzcapfed/MSci/flavour_contamination/sample_datasets/ttbar_test_mc20.h5'
logger.info("Loading ttbar h5 file")
df_ttbar = load_df(fname, split_label=split_label)
describe_df(df_ttbar, column='pt')

# Extract b-jets and c-jets with boolean array
is_c = df_ttbar["HadronGhostTruthLabelID"] == 4
is_b = df_ttbar["HadronGhostTruthLabelID"] == 5

logger.info("Extracting distributions")

contaminated_bjet_ghostpt_ttbar, true_bjet_ghostpt_ttbar, _, __ = extract_variable(df_ttbar, is_c, is_b, 'GhostBHadronsFinalPt', split_label) 
contaminated_bjet_pT_ttbar, true_bjet_pT_ttbar, contaminated_cjet_pT_ttbar, true_cjet_pT_ttbar = extract_variable(df_ttbar, is_c, is_b, "pt", split_label)
contaminated_bjet_leadingdr_ttbar, true_bjet_leadingdr_ttbar, contaminated_cjet_leadingdr_ttbar, true_cjet_leadingdr_ttbar = extract_variable(df_ttbar, is_c, is_b, 'HadronGhostTruthLabelDR', split_label)

# b-hadrdon pT / jet pT
contaminated_bpt_jetpt_ttbar = contaminated_bjet_ghostpt_ttbar / contaminated_bjet_pT_ttbar
true_bpt_jetpt_ttbar = true_bjet_ghostpt_ttbar / true_bjet_pT_ttbar


plot_path = "plots_final/"
atlas_tag = (
        r"$\mathbfit{ATLAS}$ Simulation Preliminary" + "\n"
        r"$\sqrt{s} = 13.6$ TeV" + "\n"
        r"True $t\bar{t}$ $b$-jets, 20 GeV < $p_{T}$ < 250 GeV" + "\n"
        r"Contaminated QCD $b$-jets, 20 GeV < $p_{T}$ < 3000 GeV"
    )
plot_2d_ratio_distribution(contaminated_bpt_jetpt, 
                     contaminated_bjet_leadingdr, 
                     np.concatenate((contaminated_bpt_jetpt, true_bpt_jetpt_ttbar)),
                     np.concatenate((contaminated_bjet_leadingdr, true_bjet_leadingdr_ttbar)),
                     jet_type='b',
                     x_label='$b$-hardron $p_{T}$ / jet $p_{T}$',
                     y_label='Leading hadron $\Delta R$',
                     save_fname=plot_path+'contamination.png',
                     atlas_tag=atlas_tag,
                     logx=True,
                     logy=False,
                     logz=True,
                     cut = (0.115385, 0.692308))

# plot_2d_distribution(contaminated_bpt_jetpt, 
#                      contaminated_bjet_leadingdr, 
#                      true_bpt_jetpt,
#                      true_bjet_leadingdr,
#                      jet_type='b',
#                      x_label='$b$-hardron $p_{T}$ / jet $p_{T}$',
#                      y_label='Leading hadron $\Delta R$',
#                      save_fname=plot_path+'bpt_jetpt_dr_JZ71.png',
#                      logx=True,
#                      logy=False,
#                      logz=True)

# plot_2d_distribution(contaminated_bpt_jetpt_ttbar, 
#                      contaminated_bjet_leadingdr_ttbar, 
#                      true_bpt_jetpt_ttbar,
#                      true_bjet_leadingdr_ttbar,
#                      jet_type='b',
#                      x_label='$b$-hardron $p_{T}$ / jet $p_{T}$',
#                      y_label='Leading hadron $\Delta R$',
#                      save_fname=plot_path+'bpt_jetpt_dr_ttbar.png',
#                      logx=True,
#                      logy=False,
#                      logz=True)

# ----------- TEST
atlas_tag1 = (
        r"$\mathbfit{ATLAS}$ Simulation Preliminary" + "\n"
        r"$\sqrt{s} = 13.6$ TeV, QCD events" + "\n"
        r"Contaminated $b$-jets" + "\n"
        r"20 GeV < $p_{T}$ < 3000 GeV" + "\n"
    )
atlas_tag2 = (
        r"$\mathbfit{ATLAS}$ Simulation Preliminary" + "\n"
        r"$\sqrt{s} = 13.6$ TeV, $t\bar{t}$ events" + "\n"
        r"True $b$-jets" + "\n"
        r"20 GeV < $p_{T}$ < 250 GeV" + "\n"
        ) 
plot_2d_distribution(contaminated_bpt_jetpt, 
                     contaminated_bjet_leadingdr, 
                     true_bpt_jetpt_ttbar,
                     true_bjet_leadingdr_ttbar,
                     jet_type='b',
                     x_label='$b$-hardron $p_{T}$ / jet $p_{T}$',
                     y_label='Leading hadron $\Delta R$',
                     save_fname=plot_path+'bpt_jetpt_dr_truettbar_contamJZ71.png',
                     atlas_tag1=atlas_tag1,
                     atlas_tag2=atlas_tag2,
                     logx=True,
                     logy=False,
                     logz=True)

# -----------

# plot_2d_distribution(contaminated_bpt_jetpt, 
#                      contaminated_bjet_leadingdr, 
#                      np.concatenate((contaminated_bpt_jetpt, true_bpt_jetpt_ttbar)),
#                      np.concatenate((contaminated_bjet_leadingdr, true_bjet_leadingdr_ttbar)),
#                      jet_type='b',
#                      x_label='$b$-hardron $p_{T}$ / jet $p_{T}$',
#                      y_label='Leading hadron $\Delta R$',
#                      save_fname=plot_path+'bpt_jetpt_dr_mixed_smallerbins.png',
#                      logx=True,
#                      logy=False,
#                      logz=True)

