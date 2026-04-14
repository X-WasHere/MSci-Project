"""Produce histograms for layer by layer energy deposition information"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Assuming these are in your environment based on your original script
from puma.utils import logger
import pandas as pd
from ftag.cuts import Cuts
from ftag.hdf5.h5reader import H5Reader
import glob

# ------- Functions -------
def load_data(file_pattern, num_jets=None, batch_size=500_000):
    """Loads and filters data from HDF5 files matching a pattern using H5Reader."""
    file_paths = glob.glob(file_pattern)
    if not file_paths:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    cuts = Cuts.from_list([
        ("pt", ">", 20000),
        ("pt", "<", 6000000),
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
    ])

    jet_vars = [
        "pt", "eta", "eventNumber",
        "HadronGhostTruthLabelID",
        "flavour_label"
    ]
    
    # Updated to the full 11 variables in depth-order
    calo_vars = [
        "rawE",          
        "PreSamplerB",   
        "PreSamplerE",  
        "EMB1",          
        "EME1",          
        "EMB2",         
        "EME2",         
        "TileBar0",
        "TileBar1",
        "TileBar2",     
        "TileExt0",      
        "TileGap1",     
        "HEC0",           
        "HEC1",
        "HEC2",
        "HEC3"
    ]

    variables = {"jets": jet_vars, "calo": calo_vars}

    reader = H5Reader(
        fname=file_paths,
        jets_name="jets",
        batch_size=batch_size,
        do_remove_inf=True,
        shuffle=True
    )

    data = reader.load(variables=variables, cuts=cuts, num_jets=num_jets)
    df_jets = pd.DataFrame(data["jets"])

    calo_3d = np.stack([
        data['calo']['rawE'],
        data['calo']['PreSamplerB'],
        data['calo']['PreSamplerE'],
        data['calo']['EMB1'],
        data['calo']['EME1'],
        data['calo']['EMB2'],
        data['calo']['EME2'],
        data['calo']['TileBar0'],
        data['calo']['TileBar1'],
        data['calo']['TileBar2'],
        data['calo']['TileExt0'],
        data['calo']['TileGap1'],
        data['calo']['HEC0'],
        data['calo']['HEC1'],
        data['calo']['HEC2'],
        data['calo']['HEC3'],
        ], axis=-1
    )

    return df_jets, calo_3d

def describe_df(df, column=None):
    """Prints a summary of the given dataframe to the terminal."""
    pd.set_option('display.max_columns', None)
    print("---SHAPE INFO----")
    print(df.info())
    print(df.describe())
    if column:
        print("---LOOK at desired COLUMN---")
        print(df[column].describe())
    return

def get_layer_fraction(calo_array, layer_idx, rawE_idx, mask):
    """Calculates the fraction of total energy deposited in a specific layer per jet."""
    layer_energy = np.nansum(calo_array[mask, :, layer_idx], axis=1)
    total_energy = np.nansum(calo_array[mask, :, rawE_idx], axis=1)
    
    valid_jets = total_energy > 0
    fractions = layer_energy[valid_jets] / total_energy[valid_jets]
    return fractions

# ------- Loading Data -------
fname = ""

logger.info("Loading h5 file")
df_jets, calo_3d = load_data(fname)

logger.info("Prcessing data")
is_b = df_jets['HadronGhostTruthLabelID'] == 5
is_c = df_jets['HadronGhostTruthLabelID'] == 4
is_tau = df_jets['HadronGhostTruthLabelID'] == 15
is_light = df_jets['HadronGhostTruthLabelID'] == 0

layers = {
    'PreSamplerB': 1, 'PreSamplerE': 2,
    'EMB1': 3, 'EME1': 4,
    'EMB2': 5, 'EME2': 6,
    'TileBar0': 7, 'TileBar1': 8, 'TileBar2': 9, 'TileExt0': 10, 'TileGap1': 11, 
    'HEC0': 12, 'HEC1' : 13, 'HEC2' : 14, 'HEC3' : 15
}


for layer in layers.keys():
    layer_idx = layers[layer]
    rawE_idx = 0

    # load variables
    frac_b = get_layer_fraction(calo_3d, layer_idx, rawE_idx, is_b)
    frac_c = get_layer_fraction(calo_3d, layer_idx, rawE_idx, is_c)
    frac_tau = get_layer_fraction(calo_3d, layer_idx, rawE_idx, is_tau)
    frac_light = get_layer_fraction(calo_3d, layer_idx, rawE_idx, is_light)

    # ------- Plotting -------
    logger.info("Plotting distributions")
    atlas_label = (
        r"$\mathbfit{ATLAS}$ Simulation Preliminary" + "\n"
        r"$\sqrt{s} = 13.6$ TeV" + "\n"
        r" $t\bar{t}$ events, 20 GeV < $p_{T}$ < 250 GeV" + "\n"
        r" $Z'$ events, 250 GeV < $p_{T}$ < 6 TeV"
    )

    savedir="/home/xzcapfed/MSci/GN3_calo/plots/distributions/layer_deposition_by_flavour_extended/"

    fraction_bins = np.linspace(0, 1.0, 40)
    weights_b = np.ones_like(frac_b) / len(frac_b)
    weights_c = np.ones_like(frac_c) / len(frac_c)
    weights_tau = np.ones_like(frac_tau) / len(frac_tau)
    weights_light = np.ones_like(frac_light) / len(frac_light)

    # weights_b = np.ones_like(frac_b) / is_b.sum()
    # weights_c = np.ones_like(frac_c) / is_c.sum()
    # weights_tau = np.ones_like(frac_tau) / is_tau.sum()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(frac_b, bins=fraction_bins, weights=weights_b, histtype='step', color='C0', linewidth=2)
    # ax.hist(frac_c, bins=fraction_bins, weights=weights_c, histtype='step', color='C1', linewidth=2)
    ax.hist(frac_light, bins=fraction_bins, weights=weights_light, histtype='step', color='C2', linewidth=2)
    ax.hist(frac_tau, bins=fraction_bins, weights=weights_tau, histtype='step', color='C3', linewidth=2)

    ax.plot([], [], color='C0', linewidth=2, label=r'$b$-jets')
    # ax.plot([], [], color='C1', linewidth=2, label='c-jets')
    ax.plot([], [], color='C2', linewidth=2, label='light-jets')
    ax.plot([], [], color='C3', linewidth=2, label=r'$\tau$-jets')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
    ax.tick_params(which='minor', length=3, direction='in')
    ax.tick_params(which='major', length=7, direction='in')

    ax.set_xlabel(f'Energy Fraction in {layer}', fontsize=12, loc='right')
    ax.set_ylabel('Fraction of Jets', fontsize=12, loc='top')

    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(bottom=1e-4)

    ax.legend(loc='upper right', frameon=False)

    ax.text(0.05, 0.98, atlas_label, transform=ax.transAxes, fontsize=9, verticalalignment='top')

    plt.tight_layout()
    plt.draw()
    plt.savefig(savedir + f"{layer}_fraction.png", dpi=300)
    plt.close()