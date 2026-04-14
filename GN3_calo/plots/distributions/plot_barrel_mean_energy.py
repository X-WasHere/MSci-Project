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

# ------- Loading Data -------
fname = ""

logger.info("Loading h5 file")
df_jets, calo_3d = load_data(fname)

logger.info("Prcessing data")
is_b = df_jets['HadronGhostTruthLabelID'] == 5
is_c = df_jets['HadronGhostTruthLabelID'] == 4
is_tau = df_jets['HadronGhostTruthLabelID'] == 15
is_light = df_jets['HadronGhostTruthLabelID'] == 0

# ------- Barrel Layer Analysis -------
# Barrel layers and their indices in calo_3d
barrel_layers = {
    "PreSamplerB": 1,
    "EMB1": 3,
    "EMB2": 5,
    "TileBar0": 7,
    "TileBar1": 8,
    "TileBar2": 9,
}

is_barrel = df_jets["eta"].abs() < 1.5

flavours = {
    "b-jets": is_b & is_barrel,
    "c-jets": is_c & is_barrel,
    "light-jets": is_light & is_barrel,
    "tau-jets": is_tau & is_barrel,
}

layer_names = list(barrel_layers.keys())
layer_indices = list(barrel_layers.values())
x = np.arange(len(layer_names))

fig, ax = plt.subplots(figsize=(10, 6))

colours = {"b-jets": "#1f77b4", "c-jets": "#ff7f0e", "light-jets": "#2ca02c", "tau-jets": "#d62728"}
offsets = {"b-jets": -0.15, "c-jets": -0.05, "light-jets": 0.05, "tau-jets": 0.15}

for label, mask in flavours.items():
    calo_barrel = calo_3d[mask][:, :, layer_indices] 
    energy_per_jet = calo_barrel.sum(axis=1) 
    means = energy_per_jet.mean(axis=0)
    stds = energy_per_jet.std(axis=0)

    ax.errorbar(
        x + offsets[label], means, yerr=stds,
        fmt="o", capsize=4, label=label, color=colours[label],
        markersize=5, linewidth=1.5,
    )

ax.set_xticks(x)
ax.set_xticklabels(layer_names, rotation=30, ha="right")
ax.set_xlabel("Barrel Calorimeter Layer")
ax.set_ylabel("Mean Energy Deposition [MeV]")
ax.set_title("Mean Energy Deposition per Barrel Layer (|η| < 1.5)")
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("barrel_energy_per_layer.png", dpi=150)
logger.info("Saved barrel_energy_per_layer.png")
plt.show()