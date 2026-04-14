import numpy as np
from scipy.stats import norm
import pandas as pd
import os
from datetime import datetime as dt
import matplotlib as mpl

from puma import (
    Histogram, 
    HistogramPlot, 
    VarVsVar, 
    VarVsVarPlot,
)
from puma.utils import logger
from ftag.hdf5 import H5Reader
from ftag.cuts import Cuts

from utils.reg_utils import (
    plot_median_sigma_profile,
    plot_response_binned,
)

def load_df(filepath, model_name, num_jets=None, batch_size=500_000, additional_var=None):
    """Load jets from a single HDF5 file."""

    # Define cuts as before
    cuts = Cuts.from_list([
        ("pt_visFromTruthTaus", ">", 20000),
        ("HadronGhostTruthLabelID", "==", 15), # 15 for tau
        ("eta", "<", 2.5),
        ("eta", ">", -2.5),
    ])


    jet_vars = ["pt", "eta", "HadronGhostTruthLabelID", "pt_visFromTruthTaus", f"{model_name}_pt_visFromTruthTaus", "ptFinalCalibFromTauJet"]
    if additional_var:
        jet_vars.append(additional_var)
    variables = {"jets": jet_vars}
    reader = H5Reader(
        fname=filepath,
        jets_name="jets",
        batch_size=batch_size,
        do_remove_inf=True,
        shuffle=False,
    )
    data = reader.load(variables=variables, cuts=cuts, num_jets=num_jets)
    return pd.DataFrame(data["jets"])

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

if __name__ == "__main__":

    # Temp method to combine zprime and ttbar sample
    fname_zprime='/home/xzcapfed/MSci/GN3_calo/logs/tauregress_calo/ckpts/epoch=014-val_loss=0.97745__test_zprime.h5'
    fname_ttbar='/home/xzcapfed/MSci/GN3_calo/logs/tauregress_calo/ckpts/epoch=014-val_loss=0.97745__test_ttbar.h5'

    fname_benchmark_zprime='/home/xzcapfed/MSci/GN3_calo/logs/tauregress_benchmark/ckpts/epoch=012-val_loss=0.99412__test_zprime.h5'
    fname_benchmark_ttbar='/home/xzcapfed/MSci/GN3_calo/logs/tauregress_benchmark/ckpts/epoch=012-val_loss=0.99412__test_ttbar.h5'

    model_name = "GN3_calo_tauregress"
    benchmark_name = "GN3_benchmark_tauregress"

    logger.info("Loading h5 files")
    df_ttbar = load_df(fname_ttbar, model_name)
    df_benchmark_ttbar = load_df(fname_benchmark_ttbar, benchmark_name)
    df_zprime = load_df(fname_zprime, model_name)
    df_benchmark_zprime = load_df(fname_benchmark_zprime, benchmark_name)

    describe_df(df_ttbar, 'pt')
    describe_df(df_zprime, 'pt')

    logger.info("Merging ttbar and Z' samples")
    df = pd.concat([df_ttbar, df_zprime], axis=0, ignore_index=True)
    df_benchmark = pd.concat([df_benchmark_ttbar, df_benchmark_zprime], axis=0, ignore_index=True)

    # Temp hack to plot multiple model perf - copy over column from benchmark df  
    df['GN3_benchmark_tauregress_pt_visFromTruthTaus'] = df_benchmark['GN3_benchmark_tauregress_pt_visFromTruthTaus']
    df_ttbar['GN3_benchmark_tauregress_pt_visFromTruthTaus'] = df_benchmark_ttbar['GN3_benchmark_tauregress_pt_visFromTruthTaus']
    df_zprime['GN3_benchmark_tauregress_pt_visFromTruthTaus'] = df_benchmark_zprime['GN3_benchmark_tauregress_pt_visFromTruthTaus']

    zprime_bins = [250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000]
    ttbar_bins = [20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250]

    logger.info("Plotting")
    plot_path = "/home/xzcapfed/MSci/GN3_calo/plots_tau/regression/final_no_ratios/ttbar/"
    plot_median_sigma_profile(
        df_ttbar,
        "pt_visFromTruthTaus", # TRUTH 
        # "pt", # NOTMINAL CALIBRATION 
        "ptFinalCalibFromTauJet",
        ttbar_bins,
        "$p_{T}^{truth}$",
        plot_path,
        model_names = ["GN3_calo_tauregress", "GN3_benchmark_tauregress"],
        model_label = ["GN3 calo", "GN3 benchmark"],
    )

    # plot_response_binned(
    #     df_zprime,
    #     "pt",
    #     subplot_bin_variable = "ptFromTruthDressedWZJet",
    #     subplot_bins = [20, 25, 60, 80, 100, 120, 150, 180, 210, 250],
    #     logy=True,
    #     # bins = np.arange(0.6, 1.41, 0.01),
    #     plot_filename = plot_path+f"pt_response_binnedpt_zprime.png",
    #     model_name = "GN3_calo"
    # )