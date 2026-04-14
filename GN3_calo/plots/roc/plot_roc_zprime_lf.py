"""Produce roc curves from tagger output and labels."""

from __future__ import annotations

import numpy as np

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

def get_details(df, label1, label2):
    '''Defines a boolean array to select the different flavour classes'''
    is_b = df[label1] == 5
    is_ud = (df[label1] ==0) & (df[label2] <= 2)
    is_s = (df[label1] == 0) & (df[label2] == 3)
    is_g = (df[label1] ==0) & (df[label2] == 21) # gluon jets

    n_jets_b = sum(is_b)
    n_jets_ud = sum(is_ud)
    n_jets_s = sum(is_s)
    n_jets_g = sum(is_g)

    return is_b, is_ud, is_s, is_g, n_jets_b, n_jets_ud, n_jets_s, n_jets_g


def calc_discriminant(df, name, fc, ftau, gn2=False, split_simple=False, split=False):
    if gn2:
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * df[f'{name}_pu']) + 1e-10
        disc = nom/denom
    elif split_simple:
        uscore = df[f'{name}_ps'] + df[f'{name}_pud'] + df[f'{name}_pg'] # prob of light jet is the sum of these components
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * uscore) + 1e-10
        disc = nom/denom  
    elif split:
        uscore = df[f'{name}_pghostsjets'] + df[f'{name}_pghostudjets'] + df[f'{name}_pghostgjets']
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * uscore) + 1e-10
        disc = nom/denom  
    return np.log(disc)


def load_df(file_pattern, model_name, num_jets=None, batch_size=500_000):
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
        ("pt", ">", 250000),
        ("pt", "<", 6000000),
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
        # ("eventNumber", "%10==", 9),
    ])

    # Define required variables
    jet_vars = [
        "pt", "eta", "eventNumber",
        "HadronGhostTruthLabelID",
        "flavour_label", "PartonTruthLabelID",
        f"{model_name}_pb", f"{model_name}_pc", f"{model_name}_ptau", f"{model_name}_ps",
        f"{model_name}_pud", f"{model_name}_pg"
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
    # df = df[df["HadronConeExclExtendedTruthLabelID"] == df["HadronGhostExtendedTruthLabelID"]]

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
        print("---LOOK at desired COLUMN---")
        print(df[column].describe())

    return

fname='/home/xzcapfed/MSci/GN3_calo/logs/new-data-benchmark/ckpts/epoch=004-val_loss=0.97334__test_zprime.h5'
fname_benchmark='/home/xzcapfed/MSci/GN3_calo/logs/benchmark/ckpts/epoch=003-val_loss=0.99337__test_zprime.h5'
model_name = "GN3_calo"
benchmark_name = "GN3_calo_benchmark"

# # --------- TAU TEST -----------
# fname='/home/xzcapfed/MSci/GN3_calo/logs/tauregress_calo/ckpts/epoch=014-val_loss=0.97745__test_zprime.h5'
# fname_benchmark='/home/xzcapfed/MSci/GN3_calo/logs/tauregress_benchmark/ckpts/epoch=012-val_loss=0.99412__test_zprime.h5'
# model_name = "GN3_calo_tauregress"
# benchmark_name = "GN3_benchmark_tauregress"
# # -----------------------------

logger.info("Loading h5 files")
df = load_df(fname, model_name)
df_benchmark = load_df(fname_benchmark, benchmark_name)
describe_df(df)
logger.info(f"plotting with {len(df)} jets")

logger.info("caclulate tagger discriminants")
discs_benchmark = calc_discriminant(df_benchmark, benchmark_name, fc=0.2, ftau=0.1, split_simple=True) # benchmark GN3 model
discs_calo = calc_discriminant(df, model_name, fc=0.2, ftau=0.1, split_simple=True) # new GN3 model

# defining target efficiency
sig_eff = np.linspace(0.3, 1, 100)

# defining boolean arrays to select the different flavour classes
is_b, is_ud, is_s, is_g, n_jets_b, n_jets_ud, n_jets_s, n_jets_g = get_details(df, "HadronGhostTruthLabelID", "PartonTruthLabelID")
is_b_benchmark, is_ud_benchmark, is_s_benchmark, is_g_benchmark, n_jets_b_benchmark, n_jets_ud_benchmark, n_jets_s_benchmark, n_jets_g_benchmark = get_details(df_benchmark, "HadronGhostTruthLabelID", "PartonTruthLabelID")


logger.info("Calculate rejection")
# rehection for GN3 with NO calo data
udjets_rej_benchmark = calculate_rejection(discs_benchmark[is_b_benchmark].values, discs_benchmark[is_ud_benchmark].values, sig_eff)
sjets_rej_benchmark = calculate_rejection(discs_benchmark[is_b_benchmark].values, discs_benchmark[is_s_benchmark].values, sig_eff)
gjets_rej_benchmark = calculate_rejection(discs_benchmark[is_b_benchmark].values, discs_benchmark[is_g_benchmark].values, sig_eff)
# rejection for GN3 with calo data
udjets_rej = calculate_rejection(discs_calo[is_b].values, discs_calo[is_ud].values, sig_eff)
sjets_rej = calculate_rejection(discs_calo[is_b].values, discs_calo[is_s].values, sig_eff)
gjets_rej = calculate_rejection(discs_calo[is_b].values, discs_calo[is_g].values, sig_eff)

# here the plotting of the roc starts
logger.info("Plotting ROC curves.")
plot_roc = RocPlot(
    n_ratio_panels=3,
    ylabel="Background Rejection",
    xlabel="$b$-jet Efficiency",
    atlas_second_tag="$\sqrt{s} = 13.6$ TeV, $Z'$ events\n250 GeV < $p_T$ < 6 TeV, $|\eta| < 2.5$",
    figsize=(5.5, 6),
    y_scale=1.4,
)

# Now manually assign label colours if needed
plot_roc.label_colours = {
    "GN3 benchmark": get_good_colours()[0],
    "GN3 calo": get_good_colours()[2]
}

# plot ud jet rejection 
plot_roc.add_roc(
    Roc(
        sig_eff,
        udjets_rej_benchmark,
        n_test=n_jets_ud_benchmark,
        rej_class="lquarkjets",
        signal_class="bjets",
        label="GN3 benchmark"
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        udjets_rej,
        n_test=n_jets_ud,
        rej_class="lquarkjets",
        signal_class="bjets",
        label="GN3 calo"
    ),
)


# plot s jet rejection 
plot_roc.add_roc(
    Roc(
        sig_eff,
        sjets_rej_benchmark,
        n_test=n_jets_s,
        rej_class="strangejets",
        signal_class="bjets",
        label="GN3 benchmark"
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        sjets_rej,
        n_test=n_jets_s,
        rej_class="strangejets",
        signal_class="bjets",
        label="GN3 calo"
    ),
)


# plot g rejection 
plot_roc.add_roc(
    Roc(
        sig_eff,
        gjets_rej_benchmark,
        n_test=n_jets_g_benchmark,
        rej_class="gluonjets",
        signal_class="bjets",
        label="GN3 benchmark"
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        gjets_rej,
        n_test=n_jets_g,
        rej_class="gluonjets",
        signal_class="bjets",
        label="GN3 calo"
    ),
)


# setting which flavour rejection ratio is drawn in which ratio panel
plot_roc.set_ratio_class(1, Flavours["lquarkjets"])
plot_roc.set_ratio_class(2, Flavours["gluonjets"])
plot_roc.set_ratio_class(3, Flavours["strangejets"])

plot_roc.draw()
plot_roc.savefig("/home/xzcapfed/MSci/GN3_calo/plots/roc/final/roc-zprime-calo-lf.png", transparent=False)



