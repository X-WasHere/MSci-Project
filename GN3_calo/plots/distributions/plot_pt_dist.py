"""Produce histogram of pt distribution for true vs. contaminated b-jets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import glob

from ftag import Flavours
from ftag.cuts import Cuts
from ftag.hdf5.h5reader import H5Reader

from puma import Histogram, HistogramPlot
from puma.utils import logger

def get_details(df, label):
    '''Defines a boolean array to select the different flavour classes'''
    is_light = df[label] == 0
    is_c = df[label] == 4
    is_tau = df[label] == 15
    is_b = df[label] == 5

    n_jets_b = sum(is_b)
    n_jets_light = sum(is_light)
    n_jets_c = sum(is_c)
    n_jets_tau = sum(is_tau)

    return is_light, is_c, is_b, is_tau, n_jets_light, n_jets_c, n_jets_b, n_jets_tau

def load_df(file_pattern, num_jets=None, batch_size=500_000):
    """Loads and filters data from HDF5 files matching a pattern using H5Reader."""
    file_paths = glob.glob(file_pattern)
    if not file_paths:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    # Ensure cuts match the histogram binning range
    cuts = Cuts.from_list([
        # ("pt", ">", 20000),
        # ("pt", "<", 250000),
        ("pt", ">", 250000),
        ("pt", "<", 6000000),
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
    ])

    # We only need basic variables for this plot
    jet_vars = [
        "pt", "eta", "eventNumber",
        "HadronGhostTruthLabelID",
        "flavour_label",
        "pt_btagJes",
        'HadronGhostExtendedTruthLabelID'
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

# ----------- Plotting ---------------
fname_train=''

logger.info("Loading h5 files")
df = load_df(fname_train)

logger.info(f"Loaded {len(df)} jets.")
is_light, is_c, is_b, is_tau, n_jets_light, n_jets_c, n_jets_b, n_jets_tau = get_details(df, "HadronGhostTruthLabelID")
print(f"Number light-jets : {sum(is_light)}")
print(f"Number tau-jets : {sum(is_tau)}")
print(f"Number b-jets : {sum(is_b)}")
print(f"Number c-jets : {sum(is_c)}")
print(f"Total : {sum(is_light) + sum(is_b) + sum(is_c) + sum(is_tau)}")
exit()

# Extract pt arrays and convert from MeV to GeV
pt_b = df["pt_btagJes"][is_b] / 1000.0
pt_c = df["pt_btagJes"][is_c] / 1000.0
pt_light = df["pt_btagJes"][is_light] / 1000.0

# Initialise histogram plot
plot_histo = HistogramPlot(
    n_ratio_panels=0,
    ylabel="Normalised number of jets",
    # ylabel_ratio=["Ratio"],
    xlabel="$p_T$ [GeV]",
    logy=False,
    leg_ncol=1,
    figsize=(5.5, 4.5),
    atlas_second_tag="$\\sqrt{s}=13.6$ TeV $t\\overline{t}$ events \n20 GeV < $p_T$ < 250 GeV",
    # atlas_second_tag="$\sqrt{s} = 13.6$ TeV, $Z'$ events \n250 GeV < $p_T$ < 6 TeV"
)

pt_bins = np.arange(20, 260, 10)
# pt_bins = np.arange(250, 6050, 150)
print(pt_bins)

# Add b-jets histogram
plot_histo.add(
    Histogram(
        values=pt_b,
        bins=pt_bins,
        label="$b$-jets",
        colour=Flavours["bjets"].colour,
        norm=True,
    ),
    # reference=True # Use true b-jets as the denominator for the ratio panel
)

# c-jets
plot_histo.add(
    Histogram(
        values=pt_c,
        bins=pt_bins,
        label="$c$-jets",
        colour=Flavours["cjets"].colour,
        norm=True,
    ),
)

# light-jets
plot_histo.add(
    Histogram(
        values=pt_light,
        bins=pt_bins,
        label="light-jets",
        colour=Flavours["ujets"].colour,
        norm=True,
    ),
)

plot_histo.draw()

# Save the plot
output_path = "/home/xzcapfed/MSci/GN3_calo/plots/distributions/ptbtagJes_dist_ttbar_train.png"
plot_histo.savefig(output_path, transparent=False)