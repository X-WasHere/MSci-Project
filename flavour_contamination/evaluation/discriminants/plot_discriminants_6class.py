"""Produce histogram of discriminant from tagger output and labels."""

from __future__ import annotations

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
import h5py
import glob

from ftag import Flavours
from ftag.utils import get_discriminant
from ftag.cuts import Cuts

from puma import Histogram, HistogramPlot
from puma.utils import get_good_linestyles, logger, get_dummy_2_taggers
from ftag.hdf5.h5reader import H5Reader

def get_details(df, label, seven_class):
    '''Defines a boolean array to select the different flavour classes'''
    is_light = df[label] == 0
    is_c = df[label] == 4
    is_tau = df[label] == 15
    n_jets_light = sum(is_light)
    n_jets_c = sum(is_c)
    n_jets_tau = sum(is_tau)
    
    results = {
            'is_light': is_light, 'n_jets_light': n_jets_light,
            'is_c': is_c, 'n_jets_c': n_jets_c,
            'is_tau': is_tau, 'n_jets_tau': n_jets_tau
        }

    if seven_class:
        results['is_trueb'] = (df[label] == 5) & (df['is_contaminated_bjet'] == False)
        results['is_contaminatedb'] = (df[label] == 5) & (df['is_contaminated_bjet'] == True)
        results['n_jets_trueb'] = sum(results['is_trueb'])
        results['n_jets_contaminatedb'] = sum(results['is_contaminatedb'])
    else:
        results['is_b'] = (df[label] == 5)
        results['n_jets_b'] = sum(results['is_b'])

    return results


def calc_discriminant(df, name, fc, ftau, gn2=False, split_simple=False, split_contaminated=False, split_contaminated_combined=False):
    if gn2:
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * df[f'{name}_pu']) + 1e-10
        disc = nom/denom
    elif split_simple:
        uscore = df[f'{name}_psjets'] + df[f'{name}_pudjets'] + df[f'{name}_pgjets'] # prob of light jet is the sum of these components
        nom = df[f'{name}_pb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * uscore) + 1e-10
        disc = nom/denom  
    elif split_contaminated_combined:
        uscore = df[f'{name}_ps'] + df[f'{name}_pud'] + df[f'{name}_pgluon'] 
        nom = df[f'{name}_ptrueb'] + df[f'{name}_pcontamintedb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * uscore) + 1e-10
        disc = nom/denom  
    elif split_contaminated:
        uscore = df[f'{name}_ps'] + df[f'{name}_pud'] + df[f'{name}_pgluon']
        nom = df[f'{name}_ptrueb'] + 1e-10
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + (ftau * df[f'{name}_pcontamintedb'])+ ((1-fc-ftau-ftau) * uscore) + 1e-10
        disc = nom/denom  
    return np.log(disc)


def load_df(file_pattern, model_name, num_jets=None, batch_size=500_000, seven_class=False):
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
        # ("pt", ">", 20000),
        # ("pt", "<", 250000),
        ("pt", ">", 250000),
        ("pt", "<", 6000000),
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
        # ("eventNumber", "%10==", 9),
    ])

    # Define required variables
    if seven_class:
        jet_vars = [
            "pt", "eta", "eventNumber",
            "HadronGhostTruthLabelID",
            "flavour_label", "is_contaminated_bjet",
            f"{model_name}_ptrueb", f"{model_name}_pcontamintedb", f"{model_name}_pc", f"{model_name}_ptau", f"{model_name}_ps",
            f"{model_name}_pud", f"{model_name}_pgluon"
        ]
    else:
        jet_vars = [
            "pt", "eta", "eventNumber",
            "HadronGhostTruthLabelID",
            "flavour_label",
            f"{model_name}_pb", f"{model_name}_pc", f"{model_name}_ptau", f"{model_name}_psjets",
            f"{model_name}_pudjets", f"{model_name}_pgjets"
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

fname_ttbar='/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_benchmark/ckpts/epoch=002-val_loss=0.96618__test_ttbar.h5'
fname_zprime='/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_benchmark/ckpts/epoch=002-val_loss=0.96618__test_zprime.h5'
fname_qcd='/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_benchmark/ckpts/epoch=002-val_loss=0.96618__test_qcd.h5'

fname_ttbar_benchmark='/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_6class/ckpts/epoch=003-val_loss=0.98659__test_ttbar.h5'
fname_zprime_benchmark='/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_6class/ckpts/epoch=003-val_loss=0.98659__test_zprime.h5'
fname_qcd_benchmark='/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_6class/ckpts/epoch=003-val_loss=0.98659__test_qcd.h5'

model_name = "GN3_calo_benchmark"
model_name_benchmark = "GN3_contaminated_6class"

logger.info("Loading 7-Class h5 files")
# 7 class
df_ttbar = load_df(fname_ttbar, model_name, seven_class=True)
df_zprime = load_df(fname_zprime, model_name, seven_class=True)
df_qcd = load_df(fname_qcd, model_name, seven_class=True)
df = pd.concat([df_zprime, df_qcd])
logger.info("Loading 6-Class h5 files")
# 6 class benchmark
df_ttbar_benchmark = load_df(fname_ttbar_benchmark, model_name_benchmark, seven_class=False)
df_zprime_benchmark = load_df(fname_zprime_benchmark, model_name_benchmark, seven_class=False)
df_qcd_benchmark = load_df(fname_qcd_benchmark, model_name_benchmark, seven_class=False)
df_benchmark = pd.concat([df_zprime_benchmark, df_qcd_benchmark])

describe_df(df)
describe_df(df_benchmark)
logger.info(f"plotting with {len(df)} jets")

logger.info("caclulate tagger discriminants")
discs_truebjet = calc_discriminant(df, model_name, fc=0.2, ftau=0.1, split_contaminated=True) # for true bjet efficiency 
discs_combined = calc_discriminant(df, model_name, fc=0.2, ftau=0.1, split_contaminated_combined=True) # combine true & contaminated for total bjet efficiency
discs_benchmark = calc_discriminant(df_benchmark, model_name_benchmark, fc=0.2, ftau=0.1, split_simple=True)

# defining boolean arrays to select the different flavour classes
flavours = get_details(df, "HadronGhostTruthLabelID", seven_class=True)
is_b = flavours["is_trueb"] + flavours["is_contaminatedb"]
flavours_benchmark = get_details(df_benchmark, "HadronGhostTruthLabelID", seven_class=False)

taggers = ["GN3_7Class", "GN3_6Class"]
discs = {"GN3_7Class": discs_combined,
         "GN3_6Class": discs_benchmark}
linestyles = get_good_linestyles()[:2]

bjet_discs = discs["GN3_7Class"][is_b]
working_point = 0.30
wp_cut = np.percentile(bjet_discs, (1-working_point) * 100) 

# Initialise histogram plot
plot_histo = HistogramPlot(
    n_ratio_panels=0,
    ylabel="Normalised number of jets",
    # ylabel_ratio=["Ratio to GN3 6-Class"],
    xlabel="$b$-jet discriminant",
    logy=True,
    leg_ncol=1,
    figsize=(6.5, 4.5),
    y_scale=1.5,
    # ymax_ratio=[1.5],
    # ymin_ratio=[0.5],
    # atlas_second_tag="$\\sqrt{s}=13$ TeV $t\overline{t}$ & QCD events \n20 GeV < $p_T$ < 250 GeV \n$f_{c}=0.2$, $f_{\\tau}=0.1$",
    atlas_second_tag="$\sqrt{s} = 13.6$ TeV, $Z'$ & QCD events\n250 GeV < $p_T$ < 6 TeV \n$f_{c}=0.2$, $f_{\\tau}=0.1$"
)

# Add the histograms
for tagger, linestyle in zip(taggers, linestyles):
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_b],
            bins=np.linspace(-10, 10, 25),
            label="$b$-jets" if tagger == "GN3_7Class" else None,
            colour=Flavours["bjets"].colour,
            ratio_group="bjets",
            linestyle=linestyle,
        ),
        reference=tagger=="GN3_6Class"
    )
    plot_histo.add(
        Histogram(
            values=discs[tagger][flavours["is_c"]],
            bins=np.linspace(-10, 10, 25),
            label="$c$-jets" if tagger == "GN3_7Class" else None,
            colour=Flavours["cjets"].colour,
            ratio_group="cjets",
            linestyle=linestyle,
        ),
        reference=tagger=="GN3_6Class"
    )
    plot_histo.add(
        Histogram(
            values=discs[tagger][flavours["is_light"]],
            bins=np.linspace(-10, 10, 25),
            label="$light$-jets" if tagger == "GN3_7Class" else None,
            colour=Flavours["ujets"].colour,
            ratio_group="ujets",
            linestyle=linestyle,
        ),
        reference=tagger=="GN3_6Class"
    )
    plot_histo.add(
        Histogram(
            values=discs[tagger][flavours["is_tau"]],
            bins=np.linspace(-10, 10, 25),
            label=r"$\tau$-jets" if tagger == "GN3_7Class" else None,
            colour=Flavours["taujets"].colour,
            ratio_group="taujets",
            linestyle=linestyle,
        ),
        reference=tagger=="GN3_6Class"
    )

plot_histo.draw_vlines(
    xs=[wp_cut],
    labels=[f"{working_point:.0%} WP"], 
    linestyle="dashed",              
    colour="black"                    
)

plot_histo.draw()
plot_histo.make_linestyle_legend(
    linestyles=linestyles, labels=["GN3 7-class", "GN3 6-class"], bbox_to_anchor=(0.55, 1)
)
plot_histo.savefig("/home/xzcapfed/MSci/flavour_contamination/evaluation/discriminants/b-c-l-zprime-benchmark-widerbins.png", transparent=False)