"""Produce roc curves from tagger output and labels.

This file specifically splits the light jet classification into individual 
rejections for s, ud, and gluon

"""

from __future__ import annotations

import numpy as np

from puma import Roc, RocPlot
from ftag.utils import calculate_rejection
from puma.utils import get_dummy_2_taggers, logger
import h5py
import pandas as pd
from puma.utils import get_good_colours
from ftag.cuts import Cuts
from ftag.hdf5.h5reader import H5Reader
import glob

def get_details(df, label1, label2):
    '''Defines a boolean array to select the different flavour classes.
    
    Parameters:
    df : data loaded from .h5 test file (pd.DataFrame)
    label1 : label for generating boolean array, typically "HadronGhostTruthLabelID" (str)
    label2 : second label for light jets, typically "PartonTruthLabelID" (str)

    Returns:
    Several boolean pandas series objects (pd.Series)

    '''
    is_b = df[label1] == 5
    is_light = df[label1] == 0
    is_ud = (df[label1] ==0) & (df[label2] <= 2)
    is_s = (df[label1] == 0) & (df[label2] == 3)
    is_g = (df[label1] ==0) & (df[label2] == 21) # gluon jets

    n_jets_b = sum(is_b)
    n_jets_light = sum(is_light)
    n_jets_ud = sum(is_ud)
    n_jets_s = sum(is_s)
    n_jets_g = sum(is_g)
 
    return is_b, is_light, is_ud, is_s, is_g, n_jets_b, n_jets_light, n_jets_ud, n_jets_s, n_jets_g


def calc_discriminant(df, name, fc, ftau, gn2=False, split_simple=False, split=False):
    if gn2:
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * df[f'{name}_pu']) + 1e-10
        disc = nom/denom
    elif split_simple:
        uscore = df[f'{name}_ps'] + df[f'{name}_pud'] + df[f'{name}_pg']
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
        "PartonTruthLabelID",
        "HadronGhostTruthLabelID",
        "flavour_label",
        "GN3V01_pb", "GN3V01_pc", "GN3V01_ptau", "GN3V01_pud", "GN3V01_ps", "GN3V01_pg",
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
    df : dataframe table (pd.DataFrame)
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

fname='/home/xzcapfed/tmp/ckpts/epoch=003-val_loss=1.01886__test_zprime.h5'
logger.info("Loading h5 files")
model_name = "GN3V01_smaller"
df_gn3 = load_df(fname, model_name)
# describe_df(df_gn3)
logger.info(f"plotting with {len(df_gn3)} jets")

logger.info("caclulate tagger discriminants")
discs_baseline = calc_discriminant(df_gn3, "GN3V01", fc=0.2, ftau=0.1, split_simple=True) # benchmark GN3 model
discs_smaller_gn3 = calc_discriminant(df_gn3, "GN3V01_smaller", fc=0.2, ftau=0.1, split_simple=True) # GN3 smaller model we trained

# defining target efficiency
sig_eff = np.linspace(0.6, 1, 100)

# defining boolean arrays to select the different flavour classes
is_b_gn3, is_light_gn3, is_ud_gn3, is_s_gn3, is_g_gn3, n_jets_b, n_jets_light_gn3, n_jets_ud_gn3, n_jets_s_gn3, n_jets_g_gn3 = get_details(df_gn3, "HadronGhostTruthLabelID", "PartonTruthLabelID")


logger.info("Calculate rejection")
udjets_rej_baseline = calculate_rejection(discs_baseline[is_b_gn3].values, discs_baseline[is_ud_gn3].values, sig_eff)
sjets_rej_baseline = calculate_rejection(discs_baseline[is_b_gn3].values, discs_baseline[is_s_gn3].values, sig_eff)
gjets_rej_baseline = calculate_rejection(discs_baseline[is_b_gn3].values, discs_baseline[is_g_gn3].values, sig_eff)
light_jets_rej_baseline = calculate_rejection(discs_baseline[is_b_gn3].values, discs_baseline[is_light_gn3].values, sig_eff)

# rejection for smaller GN3 trained model
udjets_rej_smaller_gn3 = calculate_rejection(discs_smaller_gn3[is_b_gn3].values, discs_smaller_gn3[is_ud_gn3].values, sig_eff)
sjets_rej_smaller_gn3 = calculate_rejection(discs_smaller_gn3[is_b_gn3].values, discs_smaller_gn3[is_s_gn3].values, sig_eff)
gjets_rej_smaller_gn3 = calculate_rejection(discs_smaller_gn3[is_b_gn3].values, discs_smaller_gn3[is_g_gn3].values, sig_eff)
light_jets_rej_smaller_gn3 = calculate_rejection(discs_smaller_gn3[is_b_gn3].values, discs_smaller_gn3[is_light_gn3].values, sig_eff)

# here the plotting of the roc starts
logger.info("Plotting ROC curves.")
plot_roc = RocPlot(
    n_ratio_panels=3,
    ylabel="Background Rejection",
    xlabel="$b$-jet Efficiency",
    # atlas_first_tag="Simulation Preliminary",
    atlas_second_tag="$\sqrt{s} = 13.6$ TeV, $Z'$ events\n250 GeV < $p_T$ < 6 TeV, $|\eta| < 2.5$",
    figsize=(5.5, 6),
    y_scale=1.4,
)

# Now manually assign label colours if needed
plot_roc.label_colours = {
    "GN3V01": get_good_colours()[0],
    "GN3V01_smaller": get_good_colours()[2]
}

# plot ud jet rejection 
plot_roc.add_roc(
    Roc(
        sig_eff,
        udjets_rej_baseline,
        n_test=n_jets_ud_gn3,
        rej_class="lquarkjets",
        signal_class="bjets",
        label="GN3V01"
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        udjets_rej_smaller_gn3,
        n_test=n_jets_ud_gn3,
        rej_class="lquarkjets",
        signal_class="bjets",
        label="GN3V01 lowstat"
    ),
)


# plot s rejection 
plot_roc.add_roc(
    Roc(
        sig_eff,
        sjets_rej_baseline,
        n_test=n_jets_s_gn3,
        rej_class="strangejets",
        signal_class="bjets",
        label="GN3V01"
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        sjets_rej_smaller_gn3,
        n_test=n_jets_s_gn3,
        rej_class="strangejets",
        signal_class="bjets",
        label="GN3V01 lowstat"
    ),
)

# plot g rejection 
plot_roc.add_roc(
    Roc(
        sig_eff,
        gjets_rej_baseline,
        n_test=n_jets_g_gn3,
        rej_class="gluonjets",
        signal_class="bjets",
        label="GN3V01"
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        gjets_rej_smaller_gn3,
        n_test=n_jets_s_gn3,
        rej_class="gluonjets",
        signal_class="bjets",
        label="GN3V01 lowstat"
    ),
)


# setting which flavour rejection ratio is drawn in which ratio panel
plot_roc.set_ratio_class(1, "lquarkjets")
plot_roc.set_ratio_class(2, "gluonjets")
plot_roc.set_ratio_class(3, "strangejets")


plot_roc.draw()
plot_roc.savefig("/home/xzcapfed/MSci/GN3_smaller_benchmark/plots/roc_zprime_light.png", transparent=False)



