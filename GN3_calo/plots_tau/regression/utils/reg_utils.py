import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from atlasify import atlasify
import os
from puma import (
    Histogram, 
    HistogramPlot,
    VarVsVar, 
    VarVsVarPlot,
) 
from utils.data_utils import standardise_jet_regression_labels
from puma.utils import get_good_colours


# Function to get mean and width of a peak in a histogrammed distribution
def get_mean_and_width(data, method='quantile_relative', weights=None):
    #weights=None # until numpy 2.0.0 works properly
    if method not in ['mean_std','quantile', 'quantile_relative', 'custom_quantile_weights']:
        raise ValueError("'method' argument must be one of 'mean_std', 'quantile','quantile_relative' or 'custom_quantile_weights'")

    # Compute mean and standard deviation
    elif method == "mean_std":
        return np.mean(data), np.std(data)

    # Compute half the central 68.2% quantile
    elif method == 'quantile':
        if len(data) == 0:
            return 0, 0
        
        quantile_method = "linear" if weights is None else "inverted_cdf"

        # median is the 50% quantile
        median = np.quantile(data, 0.5, method=quantile_method) #, weights=weights)

        # Find the -1 sigma (15.9th percentile) and +1 sigma (84.1st percentile) values,
        # which corresponds to half the central 68.2% quantile
        plus_one_sigma = np.quantile(data, 0.841, method=quantile_method) #, weights=weights)
        minus_one_sigma = np.quantile(data, 0.159, method=quantile_method) #, weights=weights)

        # Calculate the difference between +1 and -1 sigma percentiles
        central_half_quantile = (plus_one_sigma - minus_one_sigma) / 2

        return median, central_half_quantile
    
    elif method == "quantile_relative":
        if len(data) == 0:
            return 0, 0
        
        quantile_method = "linear" if weights is None else "inverted_cdf"

        # median is the 50% quantile
        median = np.quantile(data, 0.5, method=quantile_method) #, weights=weights)

        # Find the -1 sigma (15.9th percentile) and +1 sigma (84.1st percentile) values,
        # which corresponds to half the central 68.2% quantile
        plus_one_sigma = np.quantile(data, 0.841, method=quantile_method) #, weights=weights)
        minus_one_sigma = np.quantile(data, 0.159, method=quantile_method) #, weights=weights)

        # Calculate the difference between +1 and -1 sigma percentiles
        central_half_quantile = (plus_one_sigma - minus_one_sigma) / (2 * median)

        return median, central_half_quantile
    
    elif method == "custom_quantile_weights":
        values, bin_edges = np.histogram(data, weights=weights, density=True, bins=1000)
        # make histogram between min and max, 100 bins (with weights)

        bin_width = bin_edges[1]-bin_edges[0]
        minus_one_sigma = -1
        plus_one_sigma = -1
        median = -1
        running_total = 0
        for i in range(len(bin_edges)-1):
            running_total += values[i]
            if (minus_one_sigma < 0)  and (running_total > 0.159*np.sum(values)):
                minus_one_sigma = bin_edges[i]+0.5*bin_width
            if (median < 0) and (running_total > 0.5*np.sum(values)):
                median = bin_edges[i]+0.5*bin_width
            if (plus_one_sigma < 0)  and (running_total > 0.841*np.sum(values)):
                plus_one_sigma = bin_edges[i]+0.5*bin_width

        # Calculate the difference between +1 and -1 sigma percentiles
        central_half_quantile = (plus_one_sigma - minus_one_sigma) / (2 * median)

        return median, central_half_quantile


def bootstrap_uncertainties(data, unc_func, n_subsamples=10, weights=None, random_seed=42):

    def _unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    np.random.seed(random_seed)

    if weights is not None:
        shuffled_data, shuffled_weights = _unison_shuffled_copies(data, weights)
    else:
        shuffled_data = np.copy(data)
        np.random.shuffle(shuffled_data)
        shuffled_weights = None
    data_subsets = [shuffled_data[i::n_subsamples] for i in range(n_subsamples)]
    weights_subsets = [shuffled_weights[i::n_subsamples] for i in range(n_subsamples)] if weights is not None else None
    means, sigmas = [], []
    #for subset in data_subsets:
    for i in range(len(data_subsets)):
        subset = data_subsets[i] 
        subset_weights = weights_subsets[i] if weights is not None else None
        if subset_weights is not None:
            mean, sigma = unc_func(subset, weights=subset_weights)
        else:
            mean, sigma = unc_func(subset)
        means.append(mean)
        sigmas.append(sigma)
    mean_unc = np.std(means)/2   # diving by 2 for plotting error bars
    sigma_unc = np.std(sigmas)/2 # diving by 2 for plotting error bars
    return mean_unc, sigma_unc


def plot_var_dist(
    df: pd.DataFrame,
    variable: str,
    logy: bool = False,
    bins: int = 50,
    zoom_in: tuple = None,
    plot_filename:str = None,
):
    if not variable in ["mass", "pt"]:
        raise ValueError(f"variable can only be 'mass' or 'pt'")

    data_reco  = df[variable]
    data_truth = df["R10TruthLabel_R22v1_TruthGroomedJet"+variable.capitalize()]
    data_pred  = df["p_truth"+variable.capitalize()]
    dist_reco = Histogram(data_reco, label="Reconstructed")
    dist_truth = Histogram(data_truth, label="Truth")
    dist_pred = Histogram(data_pred, label="Predicted")

    xlabel_map = {"mass":"Mass", "pt":r"p$_{\rm T}$"}

    # Initialise histogram plot
    plot_histo = HistogramPlot(
        ylabel="Number of events",
        xlabel="Jet "+xlabel_map[variable]+" [GeV]",
        logy=logy,
        # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
        bins=bins,  # you can also define an integer number for the number of bins
        bins_range=zoom_in,
        norm=False,
        atlas_first_tag="Simulation Internal",
        atlas_second_tag=r"$\sqrt{s} = 13$ TeV, anti-$k_t$ R=1.0 UFO CS+SK Soft-Drop jets"+"\n"+r"$250 < p_{\rm T} < 1300$ GeV, $40 < m_J < 300$ GeV, $|\eta| < 2.0$",
        figsize=(6, 5),
        n_ratio_panels=1,
        underoverflow=True,
    )
    # Add histograms and plot
    plot_histo.add(dist_reco)
    plot_histo.add(dist_truth, reference=True)
    plot_histo.add(dist_pred)
    plot_histo.draw()
    if plot_filename:
        plot_histo.savefig(plot_filename, transparent=False)
    else:
        plot_histo.savefig("regression_tasks/variable_hists/"+variable+"_hist.png", transparent=False)


def plot_residual_dist(
    df: pd.DataFrame,
    variable: str,
    logy: bool = False,
    bins: int | list = 50,
    zoom_in: tuple = None,
    plot_filename:str = None,   
):
    if not variable in ["mass", "pt"]:
        raise ValueError(f"variable can only be 'mass' or 'pt'")

    data_truth_reco_res = df["R10TruthLabel_R22v1_TruthGroomedJet"+variable.capitalize()] - df[variable]
    data_truth_pred_res = df["R10TruthLabel_R22v1_TruthGroomedJet"+variable.capitalize()] - df["p_truth"+variable.capitalize()]
    dist_truth_reco_res = Histogram(data_truth_reco_res, label="Truth-Reconstructed")
    dist_truth_pred_res = Histogram(data_truth_pred_res, label="Truth-Predicted")

    xlabel_map = {"mass":"Mass", "pt":r"p$_{\rm T}$"}

    # Initialise histogram plot
    plot_histo = HistogramPlot(
        ylabel="Number of events",
        xlabel=xlabel_map[variable]+" residual [GeV]",
        logy=logy,
        # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
        bins=bins, #60,  # you can also define an integer number for the number of bins
        bins_range=zoom_in, #(-1e5/1e3, 2e5/1e3),  # only considered if bins is an integer
        norm=False,
        atlas_first_tag="Work in Progress",
        atlas_second_tag="$\sqrt{s} = 13$ TeV",
        figsize=(6, 5),
        n_ratio_panels=1,
        underoverflow=True,
    )
    # Add histograms and plot
    plot_histo.add(dist_truth_reco_res, reference=True)
    plot_histo.add(dist_truth_pred_res)
    plot_histo.axis_top.axvline(0, c="k", ls=":", alpha=0.5)
    plot_histo.draw()
    if plot_filename:
        plot_histo.savefig(plot_filename, transparent=False)
    else:
        plot_histo.savefig("regression_tasks/residuals/"+variable+"_residual_hist.png", transparent=False)


def plot_abs_residual_profile(
    df: pd.DataFrame,
    res_variable: str,
    x_variable_column: str,
    x_bins: list | int = None,
    equal_pop_bins: bool = False,
    plot_xlim: tuple = None,
    plot_ylim: tuple = None,
    plot_filename: str = None,
):
    if not res_variable in ["mass", "pt"]:
        raise ValueError(f"variable can only be 'mass' or 'pt'")

    data_truth_reco_abs_res = np.abs(df["R10TruthLabel_R22v1_TruthGroomedJet"+res_variable.capitalize()] - df[res_variable])
    data_truth_pred_abs_res = np.abs(df["R10TruthLabel_R22v1_TruthGroomedJet"+res_variable.capitalize()] - df["p_truth"+res_variable.capitalize()])

    x_plot = df[x_variable_column]
    
    #TODO: x_bins, xlim, ylim, xylabels, savefilename

    res_var_map = {"mass":"Mass", "pt":"$p_T$"}
    x_var_map = {
        "mass": "Reconstructed Mass",
        "pt": r"Reconstructed p$_{\rm T}$", 
        "R10TruthLabel_R22v1_TruthGroomedJetMass": "Truth Mass", 
        "R10TruthLabel_R22v1_TruthGroomedJetPt": r"Truth p$_{\rm T}$",
        "p_truthMass": "Predicted Mass",
        "p_truthPt": r"Predicted p$_{\rm T}$"
    }
    savefile_map = {
        "mass": "mass", 
        "pt": "pt", 
        "R10TruthLabel_R22v1_TruthGroomedJetMass": "truthMass",
        "R10TruthLabel_R22v1_TruthGroomedJetPt": "truthPt",
        "p_truthMass": "predMass",
        "p_truthPt": "predPt"
    }

    if equal_pop_bins:
        if type(x_bins) != int:
            raise ValueError(f"equal_pop_bins is True, x_bins must be an integer")
        bins = pd.qcut(df[x_variable_column], q=x_bins, labels=False, duplicates='drop')
        # Compute the bin edges
        x_bins = df[x_variable_column].groupby(bins).agg(['max'])
        print(x_bins)
        x_bins = np.ravel(x_bins.values.tolist())
        print(f"x_bins are {x_bins}")

        # Calculate bin widths
        bin_widths = [x_bins[i+1] - x_bins[i] for i in range(len(x_bins) - 1)]
        print(len(bin_widths))  # Check the length of bin_widths
        print(len(x_plot))  # Check the length of x_plot
        print(len(data_truth_reco_abs_res))  # Check the length of data_truth_reco_abs_res

    plt.figure()
    sns.regplot(x=x_plot, y=data_truth_reco_abs_res, x_bins=x_bins, fit_reg=False, label="Truth-Reconstructed", color="m", marker="_", x_ci=95)#, scatter_kws={'s': bin_widths})
    sns.regplot(x=x_plot, y=data_truth_pred_abs_res, x_bins=x_bins, fit_reg=False, label="Truth-Predicted", color="g", marker="_", x_ci=95)#, scatter_kws={'s': bin_widths})
    plt.legend(frameon=False)
    plt.xlabel("Jet "+x_var_map[x_variable_column]+" [GeV]")
    plt.ylabel("abs. "+res_var_map[res_variable]+" Residual [GeV]")
    if plot_xlim:
        plt.xlim(plot_xlim)
    if plot_ylim:
        plt.ylim(plot_ylim)

    if plot_filename:
        plt.savefig(plot_filename, transparent=False)
    else:
        plt.savefig("regression_tasks/residuals/"+res_variable+"_residual_profile_vs_"+savefile_map[x_variable_column]+".png", transparent=False)


def plot_response_overall(
    df: pd.DataFrame,
    variable: str,
    mean_std_method: str = "quantile_relative",
    weights=None,
    logy: bool = False,
    bins: int | list = 50,
    bins_range: tuple = None,
    manual_range_text: str = None,
    save_stats: bool = False,
    plot_filename: str = None,
):
    if not variable in ["mass", "pt"]:
        raise ValueError(f"variable can only be 'mass' or 'pt'")

    #weights=None # until numpy 2.0.0 works properly
    
    data_reco  = df[variable]
    data_truth = df["R10TruthLabel_R22v1_TruthGroomedJet"+variable.capitalize()]
    data_pred  = df["p_truth"+variable.capitalize()]
    reco_response = Histogram(data_reco / data_truth, weights=weights, label = "Nominal Calibration")
    pred_response = Histogram(data_pred / data_truth, weights=weights, label = r"Large-$R$ Regression")

    reco_response.linestyle = "dashed"

    if not manual_range_text:
        #reco_mass_range = [int(df["mass"].min()), int(df["mass"].max())]
        #reco_pt_range = [int(df["pt"].min()), int(df["pt"].max())]
        #manual_range_text = r"$\sqrt{s} = 13$ TeV"+"\n"+str(reco_pt_range[0])+r"$ < p_{\rm T} < $"+str(reco_pt_range[1])+" GeV\n"+str(reco_mass_range[0])+r"$ < m_J < $"+str(reco_mass_range[1])+" GeV"
        #manual_range_text = r"$\sqrt{s} = 13$ TeV"+"\n"+r"$250 < p_{\rm T} < 1300$ GeV"+"\n"+r"$40 < m_J < 300$ GeV"
        #manual_range_text = r"$\sqrt{s} = 13$ TeV, anti-$k_t$ R=1.0 UFO CS+SK Soft-Drop jets"+"\n"+r"$250 < p_{\rm T} < 1300$ GeV, $40 < m_J < 300$ GeV, $|\eta| < 2.0$"
        manual_range_text = r"$\sqrt{s} = 13$ TeV, anti-$k_t$ R=1.0 UFO CS+SK Soft-Drop jets"+"\n"+r"$250 < p_{\rm T} < 1300$ GeV"+"\n"+r"$40 < m_J < 300$ GeV"

    label_map = {"mass":r"m$_{\rm Reco}$/m$_{\rm Truth}$", "pt":r"p$_{\rm T}^{\rm Reco}$/p$_{\rm T}^{\rm Truth}$"}

    reco_mean, reco_sigma = get_mean_and_width(data_reco / data_truth, method=mean_std_method, weights=weights)
    pred_mean, pred_sigma = get_mean_and_width(data_pred / data_truth, method=mean_std_method, weights=weights)

    print()
    print(f"Variable: {variable}")
    if mean_std_method == "mean_std":
        print(f"\tReco: mean = {reco_mean:.3f}, std = {reco_sigma:.3f}")
        print(f"\tPred: mean = {pred_mean:.3f}, std = {pred_sigma:.3f}")
    elif mean_std_method == "quantile":
        print(f"\tReco: median = {reco_mean:.3f}, mid 68%/2 = {reco_sigma:.3f}")
        print(f"\tPred: median = {pred_mean:.3f}, mid 68%/2 = {pred_sigma:.3f}")

    if pred_sigma-reco_sigma > 0:
        print(f"\tPredicted peak is {100*(pred_sigma - reco_sigma)/reco_sigma:.2f}% wider")
    else:
        print(f"\tPredicted peak is {100*(reco_sigma - pred_sigma)/reco_sigma:.2f}% narrower")
    
    pred_bias = abs(pred_mean-1)
    reco_bias = abs(reco_mean-1)
    if pred_bias < reco_bias:
        print(f"\tPredicted peak is {100*(reco_bias-pred_bias):.2f}% closer to 1")
    else:
        print(f"\tPredicted peak is {100*(pred_bias-reco_bias):.2f}% further from 1")
    
    if save_stats:
        if plot_filename:
            plot_folder = "/".join(plot_filename.split("/")[:-1])+"/"
        with open(plot_folder+"response_stats.txt", "a+") as f:
            f.write(f"Variable: {variable}\n")
            if mean_std_method == "mean_std":
                f.write(f"\tReco: mean = {reco_mean:.3f}, std = {reco_sigma:.3f}\n")
                f.write(f"\tPred: mean = {pred_mean:.3f}, std = {pred_sigma:.3f}\n")
            elif mean_std_method == "quantile":
                f.write(f"\tReco: median = {reco_mean:.3f}, mid 68%/2 = {reco_sigma:.3f}\n")
                f.write(f"\tPred: median = {pred_mean:.3f}, mid 68%/2 = {pred_sigma:.3f}\n")
            if pred_sigma-reco_sigma > 0:
                f.write(f"\tPredicted peak is {100*(pred_sigma - reco_sigma)/reco_sigma:.2f}% wider\n")
            else:
                f.write(f"\tPredicted peak is {100*(reco_sigma - pred_sigma)/reco_sigma:.2f}% narrower\n")
            if pred_bias < reco_bias:
                f.write(f"\tPredicted peak is {100*(reco_bias-pred_bias):.2f}% closer to 1\n")
            else:
                f.write(f"\tPredicted peak is {100*(pred_bias-reco_bias):.2f}% further from 1\n")

    # Initialise histogram plot
    plot_histo = HistogramPlot(
        ylabel="a.u.",
        xlabel=label_map[variable],
        logy=logy,
        # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
        bins=bins, #60,  # you can also define an integer number for the number of bins
        bins_range=bins_range, #(-1e5/1e3, 2e5/1e3),  # only considered if bins is an integer
        norm=True,
        atlas_first_tag="Simulation Preliminary",
        atlas_second_tag=manual_range_text, #r"$\sqrt{s} = 13$ TeV"+"\n"+r"$250 < p_{\rm T} < 1300$ GeV"+"\n"+r"$50 < m_J < 300$ GeV",
        #atlas_tag_outside=True,
        figsize=(6, 5),
        n_ratio_panels=1,
        underoverflow=False,
    )
    # Add histograms and plot
    plot_histo.add(reco_response, reference=True)
    plot_histo.add(pred_response)
    plot_histo.axis_top.axvline(1, c="k", ls=":", alpha=0.5)
    plot_histo.draw()
    if plot_filename:
        plot_histo.savefig(plot_filename, transparent=False)
    else:
        plot_histo.savefig(variable+"_response_hist.png", transparent=False)
    

def plot_response_binned(
    df: pd.DataFrame,
    variable: str,
    subplot_bin_variable: str,
    subplot_bins: list = [250, 300, 400, 500, 600, 700, 800, 900, 1000, 1300],
    mean_std_method: str = "quantile",
    edge_bins: bool = False,
    logy: bool = False,
    bins: int | list = 50,
    bins_plot_range: tuple = None,
    plot_filename:str = None,
    model_name: str = None,
):
    if not variable in ["mass", "pt"]:
        raise ValueError(f"variable can only be 'mass' or 'pt'")

    #if not len(subplot_bins) == 10:
    #    raise ValueError(f"only 9 subplots are supported, len(subplot_bins) must be 10")

    #if not ((len(subplot_bins) == 10 and not edge_bins) or (len(subplot_bins) == 8 and edge_bins)):
    #    raise ValueError(f"len(subplot_bins) must be 10 if edge_bins is False and 8 if edge_bins is True")
    
    if edge_bins:
        #subplot_bins.insert(0, float('-inf'))
        subplot_bins.append(float('inf'))

    variable_map = {
        # "pt":r"p$_{\rm T}^{\rm Reco}$",
        "ptFromTruthDressedWZJet":r"p$_{\rm T}^{\rm Truth}$",
        # "pt_visFromTruthTaus":r"p$_{\rm T}^{\rm Truth}$",
        # "GN3_v00_ptFromTruthJet":r"p$_{\rm T}^{\rm Pred}$"
        f"{model_name}_ptFromTruthDressedWZJet":r"p$_{\rm T}^{\rm Pred}$"
        # "GN3_v00_lowstat_lowpt_pt":r"p$_{\rm T}^{\rm Pred}$"
    }

    label_map = {
        "pt":r"p$_{\rm T}^{\rm Reco}$/p$_{\rm T}^{\rm Truth}$"
    }

    legend_map = {
        "pt_reco": r"Nominal Calibration",   # r"Reco level p$_{\rm T}$",
        "pt_pred": r"Model Prediction",              # r"p$_{\rm T}^{GN2X}$ Regression",
    }

    # create figure with 9 subfigures
    fig, axs = plt.subplots(3, 3, figsize=(12, 9), sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})

    # loop over subfigures
    for i in range(len(subplot_bins)-1):
        subplot_mask = \
            (df[subplot_bin_variable]/1000 >= subplot_bins[i]) & \
            (df[subplot_bin_variable]/1000 < subplot_bins[i+1])

        data_reco = df[subplot_mask]["pt"] / 1000
        data_truth = df[subplot_mask]["ptFromTruthDressedWZJet"] / 1000
        data_pred = df[subplot_mask][f"{model_name}_ptFromTruthDressedWZJet"] / 1000

        valid_mask = ~data_truth.isna()
        reco_response = data_reco[valid_mask] / data_truth[valid_mask]
        pred_response = data_pred[valid_mask] / data_truth[valid_mask]

        from scipy.stats import kurtosis
        reco_kurtosis = kurtosis(data_reco)
        pred_kurtosis = kurtosis(data_pred)
        
        reco_mean, reco_sigma = get_mean_and_width(reco_response, method=mean_std_method)
        pred_mean, pred_sigma = get_mean_and_width(pred_response, method=mean_std_method)

        ax = axs[i//3, i%3]
        ax.hist(reco_response, bins=bins, range=bins_plot_range, histtype="step", label=legend_map[variable+"_reco"], density=False, color="C4")
        ax.hist(pred_response, bins=bins, range=bins_plot_range, histtype="step", label=legend_map[variable+"_pred"], density=False, color="C2")        
        ax.axvline(1, c="k", ls=":", alpha=0.5)
        ax.set_yscale("log") if logy else None

        # Only set y label for leftmost subplots
        if i >= 6:
            ax.set_xlabel(label_map[variable])
        if i % 3 == 0:
            ax.set_ylabel("Number of jets")

        #add text with bin range
        if edge_bins and i == 0:
            bin_range_text = f"{subplot_bins[i]} < {variable_map[subplot_bin_variable]} [GeV] < {subplot_bins[i+1]}"
            #bin_range_text = f"{variable_map[subplot_bin_variable]} [GeV] < {subplot_bins[i+1]}"
        elif edge_bins and i == len(subplot_bins)-2:
            bin_range_text = f"{variable_map[subplot_bin_variable]} [GeV] > {subplot_bins[i]}"
        else:
            bin_range_text = f"{subplot_bins[i]} < {variable_map[subplot_bin_variable]} [GeV] < {subplot_bins[i+1]}"

        ax.text(0.97, 0.97, bin_range_text,
            transform = ax.transAxes,
            verticalalignment='top', 
            horizontalalignment='right',
            fontsize=9)
        
        # ax.set_ylim(0, 2500)
        
        #add text with mean, std
        if mean_std_method == "mean_std":
            ax.text(0.97, 0.86, f"(mean, std, kurtosis)",
                transform = ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=9)
        elif mean_std_method == "quantile":
            ax.text(0.97, 0.86, f"(med, mid 68%/2, kurtosis)",
                transform = ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=9)
        ax.text(0.97, 0.80, f"({reco_mean:.2f}, {reco_sigma:.2f}, {reco_kurtosis:.2f})",
            transform = ax.transAxes,
            verticalalignment='top', horizontalalignment='right', color="C4",
            fontsize=9)
        ax.text(0.97, 0.74, f"({pred_mean:.2f}, {pred_sigma:.2f}, {pred_kurtosis:.2f})",
            transform = ax.transAxes,
            verticalalignment='top', horizontalalignment='right', color="C2",
            fontsize=9)
        
        # plot text with % of total jets
        percentage_jets = len(df[subplot_mask])/len(df)
        ax.text(0.03, 0.96, f"{percentage_jets*100:.1f}% of jets",
                            transform = ax.transAxes,
                            verticalalignment='top', horizontalalignment='left',
                            fontsize = 9, alpha = 0.5)

    
    # Add legend outside of the plots on the top right
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.98), fontsize='large', frameon=False)

    #atlas_font = {'fontname':'Open Sans'}
    #fig.text(0.1, 0.9, r"$\bf{\it{ATLAS}}$ Work in Progress", 
    #         transform = fig.transFigure, ha='left', fontsize='large', fontname='Nimbus Sans')

    if plot_filename:
        fig.savefig(plot_filename, transparent=False)
    else:
        fig.savefig(variable+"_response_binned.png", transparent=False)
    

        
def plot_median_sigma_profile(
    df: pd.DataFrame,
    x_variable: str,
    y_variable: str,
    x_bins: int | list = 10,
    x_label: str = None,
    plot_folder: str = None,
    model_names: list[str] = None,
    model_label: list[str] = None,
):

    label_map = {
        "mass":r"$m$", 
        "pt":r"$p_{\rm T}$",
        "ptFinalCalibFromTauJet":r"$p_{\rm T}$"
    }

    if not y_variable in ["mass", "pt", "ptFinalCalibFromTauJet"]:
        raise ValueError(f"variable can only be 'mass' or 'pt'")
    if isinstance(x_bins, int):
        x_bins = np.linspace(np.min(df[x_variable]), np.max(df[x_variable]), x_bins)

    reco_medians, reco_sigmas = [], []
    reco_medians_unc, reco_sigmas_unc = [], []


    pred_medians_dict = {name: [] for name in model_names}
    pred_sigmas_dict = {name: [] for name in model_names}
    pred_medians_unc_dict = {name: [] for name in model_names}
    pred_sigmas_unc_dict = {name: [] for name in model_names}

    for i in range(len(x_bins)-1):
        subplot_mask = \
            (df[x_variable]/1000 >= x_bins[i]) & \
            (df[x_variable]/1000 < x_bins[i+1])

        data_reco  = df[subplot_mask][y_variable]
        data_truth = df[subplot_mask][x_variable]

        valid_mask = ~data_truth.isna()
        response_reco = data_reco[valid_mask] / data_truth[valid_mask]


        median_reco, sigma_reco = get_mean_and_width(response_reco, method="quantile_relative")
        median_reco_unc, sigma_reco_unc = bootstrap_uncertainties(response_reco, get_mean_and_width, n_subsamples=10)

        reco_medians.append(median_reco)
        reco_sigmas.append(sigma_reco)
        reco_medians_unc.append(median_reco_unc)
        reco_sigmas_unc.append(sigma_reco_unc)

        for i, name in enumerate(model_names):
            data_pred = df[subplot_mask][f'{name}_{x_variable}']
            response_pred = data_pred[valid_mask] / data_truth[valid_mask]
            
            median_pred, sigma_pred = get_mean_and_width(response_pred, method="quantile_relative")
            median_pred_unc, sigma_pred_unc = bootstrap_uncertainties(response_pred, get_mean_and_width, n_subsamples=10)

            pred_medians_dict[name].append(median_pred)
            pred_sigmas_dict[name].append(sigma_pred)
            pred_medians_unc_dict[name].append(median_pred_unc)
            pred_sigmas_unc_dict[name].append(sigma_pred_unc)


    bin_centres = (np.array(x_bins[:-1]) + np.array(x_bins[1:]))/2
    bin_widths = np.array(x_bins[1:]) - np.array(x_bins[:-1])

    response_median_reco = VarVsVar(bin_centres, reco_medians, reco_medians_unc, x_var_widths=bin_widths, plot_y_std=False, label="Nominal Calibration")
    response_sigma_reco  = VarVsVar(bin_centres, reco_sigmas, reco_sigmas_unc, x_var_widths=bin_widths, plot_y_std=False, label="Nominal Calibration")
    response_median_reco.linestyle = "dashed"
    response_sigma_reco.linestyle = "dashed"

    if max(x_bins) > 2000:
        second_tag = r"$\sqrt{s} = 13.6$ TeV, $Z'$ events" + "\n250 GeV < Truth $p_T$ < 6 TeV , $|\eta| < 2.5$"
    else: 
        second_tag = r"$\sqrt{s} = 13.6$ TeV, $t\overline{t}$ events" + "\n20 GeV < Truth $p_T$ < 250 GeV , $|\eta| < 2.5$"

    tau_char = r"$\tau$"
    b_char = "b"
    plot_varVsVar = VarVsVarPlot(
        ylabel = f"Median {tau_char if 'Tau' in x_variable else b_char}-jet {label_map[y_variable]} Response",
        xlabel=x_label,
        # ylabel_ratio=["Signed RSD"],
        atlas_first_tag="Simulation Internal",
        atlas_second_tag = second_tag,
        logy=False,
        figsize=(5.5, 4.5),
        n_ratio_panels=0,
        ymin=0.94,
        ymax=1.06,
        dpi=400
        #ratio_method = "root_square_diff",
    )
    # plot_varVsVar.add(response_median_reco, reference=False)

    for i, name in enumerate(model_names):
        response_median_pred = VarVsVar(
            bin_centres,
            pred_medians_dict[name],
            pred_medians_unc_dict[name],
            x_var_widths=bin_widths,
            plot_y_std=False,
            label=f"{model_label[i]}",
            colour=f"{'tab:blue' if 'benchmark' in name else 'firebrick'}"
        )
        is_benchmark = "benchmark" in name.lower()
        plot_varVsVar.add(response_median_pred, reference=False)

    # unity_reference = VarVsVar(
    #     bin_centres, 
    #     np.ones_like(bin_centres), 
    #     np.zeros_like(bin_centres), 
    #     x_var_widths=bin_widths, 
    #     plot_y_std=False, 
    #     label="Reference"
    # )
    # unity_reference.color = "grey" 
    # unity_reference.alpha = 0.5

    # plot_varVsVar.add(unity_reference, reference=True)

    plot_varVsVar.axis_top.axhline(1, c="k", ls=":", alpha=0.5)
    plot_varVsVar.draw()
    if plot_folder:
        plot_varVsVar.savefig(plot_folder+f"{y_variable}_response_median_vs_{x_variable}_preliminary.png", transparent=False)
        # plot_varVsVar.savefig(plot_folder+f"{y_variable}_response_median_vs_{x_variable}_preliminary.pdf", transparent=False)
    else:
        plot_varVsVar.savefig(f"{y_variable}_response_median_vs_{x_variable}.png", transparent=False)

    plot_varVsVar = VarVsVarPlot(
        ylabel=r"Relative $\tau$-jet "+label_map[y_variable]+" Resolution",
        xlabel=x_label,
        # ylabel_ratio=["RSD"],
        atlas_first_tag="Simulation Internal",
        atlas_second_tag=second_tag,
        logy=False,
        figsize=(6, 5),
        n_ratio_panels=1,
        # ratio_method = "root_square_diff",
        dpi=400
    )
    # plot_varVsVar.add(response_sigma_reco, reference=False)
    for i, name in enumerate(model_names):
        response_sigma_pred = VarVsVar(
            bin_centres,
            pred_sigmas_dict[name],
            pred_sigmas_unc_dict[name],
            x_var_widths=bin_widths,
            plot_y_std=False,
            label=f"{model_label[i]}",
            colour=f"{'tab:blue' if 'benchmark' in name else 'firebrick'}"
        )
        is_benchmark = "benchmark" in name.lower()
        plot_varVsVar.add(response_sigma_pred, reference=is_benchmark)

    plot_varVsVar.draw()
    if plot_folder:
        plot_varVsVar.savefig(plot_folder+f"{y_variable}_response_resolution_vs_{x_variable}_preliminary.png", transparent=False)
        # plot_varVsVar.savefig(plot_folder+f"{y_variable}_response_resolution_vs_{x_variable}_preliminary.pdf", transparent=False)
    else:
        plot_varVsVar.savefig(f"{y_variable}_response_resolution_vs_{x_variable}.png", transparent=False)


        
def plot_median_sigma_profile_only_pt(
    df: pd.DataFrame,
    x_variable: str,
    y_variable: str,
    x_bins: int | list = 10,
    x_label: str = None,
    plot_folder: str = None,
):

    label_map = {
        "mass":r"$m$", 
        "pt":r"$p_{\rm T}$"
    }

    if not y_variable in ["mass", "pt", "ptFinalCalibFromTauJets"]:
        raise ValueError(f"variable can only be 'mass' or 'pt'")
    if isinstance(x_bins, int):
        x_bins = np.linspace(np.min(df[x_variable]), np.max(df[x_variable]), x_bins)

    reco_medians, reco_sigmas = [], []
    reco_medians_unc, reco_sigmas_unc = [], []

    for i in range(len(x_bins)-1):
        subplot_mask = \
            (df[x_variable]/1000 >= x_bins[i]) & \
            (df[x_variable]/1000 < x_bins[i+1])

        data_reco  = df[subplot_mask][y_variable]
        # data_truth = df[subplot_mask]["R10TruthLabel_R22v1_TruthGroomedJet"+y_variable.capitalize()]
        data_truth = df[subplot_mask][x_variable]
        # data_pred  = df[subplot_mask]["p_truth"+y_variable.capitalize()]

        valid_mask = ~data_truth.isna()
        response_reco = data_reco[valid_mask] / data_truth[valid_mask]
        # response_reco = data_reco / data_truth
        # response_pred = data_pred / data_truth

        median_reco, sigma_reco = get_mean_and_width(response_reco, method="quantile_relative")

        reco_medians.append(median_reco)
        reco_sigmas.append(sigma_reco)
        
        median_reco_unc, sigma_reco_unc = bootstrap_uncertainties(response_reco, get_mean_and_width, n_subsamples=10)

        reco_medians_unc.append(median_reco_unc)
        reco_sigmas_unc.append(sigma_reco_unc)

    bin_centres = (np.array(x_bins[:-1]) + np.array(x_bins[1:]))/2
    bin_widths = np.array(x_bins[1:]) - np.array(x_bins[:-1])

    response_median_reco = VarVsVar(bin_centres, reco_medians, reco_medians_unc, x_var_widths=bin_widths, plot_y_std=False, label="Nominal Calibration")
    response_sigma_reco  = VarVsVar(bin_centres, reco_sigmas, reco_sigmas_unc, x_var_widths=bin_widths, plot_y_std=False, label="Nominal Calibration")
    response_median_reco.linestyle = "dashed"
    response_sigma_reco.linestyle = "dashed"



    plot_varVsVar = VarVsVarPlot(
        ylabel=r"Median $b$-jet "+label_map[y_variable]+" Response",
        xlabel=x_label,
        #ylabel_ratio=["RSD"],
        atlas_first_tag="Simulation Internal",
        atlas_second_tag=r"$t\overline{t}$ & $Z'$ events $p_T$_{Truth} > 20 GeV, $|\eta| < 2.5$",
        # atlas_second_tag=r"$Z'$ 250 < $p_T$ < 6000 GeV, $|\eta| < 2.5$",
        #atlas_tag_outside=True,
        logy=False,
        figsize=(6, 5),
        n_ratio_panels=1,
        # ymin=0.9,
        # ymax=1.2,
        ratio_method = "root_square_diff",
    )
    plot_varVsVar.add(response_median_reco, reference=True)
    plot_varVsVar.axis_top.axhline(1, c="k", ls=":", alpha=0.5)
    plot_varVsVar.draw()
    if plot_folder:
        plot_varVsVar.savefig(plot_folder+f"{y_variable}_response_median_vs_{x_variable}.png", transparent=False)
    else:
        plot_varVsVar.savefig(f"{y_variable}_response_median_vs_{x_variable}.png", transparent=False)

    plot_varVsVar = VarVsVarPlot(
        ylabel=r"Relative $b$-jet "+label_map[y_variable]+" Resolution",
        xlabel=x_label,
        ylabel_ratio=["RSD"],
        atlas_first_tag="Simulation Internal",
        atlas_second_tag=r"$t\overline{t}$ & $Z'$ ${p_T}_{Truth}$ > 20 GeV, $|\eta| < 2.5$",
        # atlas_second_tag=r"$Z'$ 250 < $p_T$ < 6000 GeV, $|\eta| < 2.5$",
        #atlas_tag_outside=True,
        logy=False,
        figsize=(6, 5),
        n_ratio_panels=1,
        ratio_method = "root_square_diff",
    )
    plot_varVsVar.add(response_sigma_reco, reference=True)
    plot_varVsVar.draw()
    if plot_folder:
        plot_varVsVar.savefig(plot_folder+f"{y_variable}_response_resolution_vs_{x_variable}.png", transparent=False)
    else:
        plot_varVsVar.savefig(f"{y_variable}_response_resolution_vs_{x_variable}.png", transparent=False)



        
def plot_median_sigma_profile_multiple(
    df: pd.DataFrame,
    x_variable: str,
    y_variable: str,
    x_bins: int | list = 10,
    x_label: str = None,
    plot_folder: str = None,
    model_names: list[str] = None,
    model_label: list[str] = None,
    model_df: list[str] = None,
    ratio_mine: bool = False,
):

    label_map = {
        "mass":r"$m$", 
        "pt":r"$p_{\rm T}$"
    }

    if not y_variable in ["mass", "pt", "ptFinalCalibFromTauJets"]:
        raise ValueError(f"variable can only be 'mass' or 'pt'")
    if isinstance(x_bins, int):
        x_bins = np.linspace(np.min(df[x_variable]), np.max(df[x_variable]), x_bins)

    reco_medians, reco_sigmas = [], []
    reco_medians_unc, reco_sigmas_unc = [], []

    if model_df is not None:
        pred_medians_dict = {}
        pred_sigmas_dict = {}
        pred_medians_unc_dict = {}
        pred_sigmas_unc_dict = {}

        for i, name in enumerate(model_names):
            key = name + model_df[i]  # e.g. "ModelA_nominal"
            pred_medians_dict[key] = []
            pred_sigmas_dict[key] = []
            pred_medians_unc_dict[key] = []
            pred_sigmas_unc_dict[key] = []
    else:
        pred_medians_dict = {name: [] for name in model_names}
        pred_sigmas_dict = {name: [] for name in model_names}
        pred_medians_unc_dict = {name: [] for name in model_names}
        pred_sigmas_unc_dict = {name: [] for name in model_names}

    for i in range(len(x_bins)-1):
        subplot_mask = \
            (df[x_variable]/1000 >= x_bins[i]) & \
            (df[x_variable]/1000 < x_bins[i+1])

        data_reco  = df[subplot_mask][y_variable]
        # data_truth = df[subplot_mask]["R10TruthLabel_R22v1_TruthGroomedJet"+y_variable.capitalize()]
        data_truth = df[subplot_mask][x_variable]
        # data_pred  = df[subplot_mask]["p_truth"+y_variable.capitalize()]

        valid_mask = ~data_truth.isna()
        response_reco = data_reco[valid_mask] / data_truth[valid_mask]
        # response_reco = data_reco / data_truth
        # response_pred = data_pred / data_truth

        median_reco, sigma_reco = get_mean_and_width(response_reco, method="quantile_relative")

        reco_medians.append(median_reco)
        reco_sigmas.append(sigma_reco)
        
        median_reco_unc, sigma_reco_unc = bootstrap_uncertainties(response_reco, get_mean_and_width, n_subsamples=10)

        reco_medians_unc.append(median_reco_unc)
        reco_sigmas_unc.append(sigma_reco_unc)

        for i, name in enumerate(model_names):
            if model_df is not None:
                data_pred = df[subplot_mask][f'{name}']
            else:
                data_pred = df[subplot_mask][f'{name}']
            response_pred = data_pred[valid_mask] / data_truth[valid_mask]

            median_pred, sigma_pred = get_mean_and_width(response_pred, method="quantile_relative")
            median_pred_unc, sigma_pred_unc = bootstrap_uncertainties(response_pred, get_mean_and_width, n_subsamples=10)

            pred_medians_dict[name + model_df[i]].append(median_pred)
            pred_sigmas_dict[name + model_df[i]].append(sigma_pred)
            pred_medians_unc_dict[name + model_df[i]].append(median_pred_unc)
            pred_sigmas_unc_dict[name + model_df[i]].append(sigma_pred_unc)



    bin_centres = (np.array(x_bins[:-1]) + np.array(x_bins[1:]))/2
    bin_widths = np.array(x_bins[1:]) - np.array(x_bins[:-1])

    response_median_reco = VarVsVar(bin_centres, reco_medians, reco_medians_unc, x_var_widths=bin_widths, plot_y_std=False, label="Nominal Calibration")
    response_sigma_reco  = VarVsVar(bin_centres, reco_sigmas, reco_sigmas_unc, x_var_widths=bin_widths, plot_y_std=False, label="Nominal Calibration")
    response_median_reco.linestyle = "dashed"
    response_sigma_reco.linestyle = "dashed"

    second_tag = r"$\sqrt{s} = 13$ TeV, $t\overline{t}$"
    second_tag += "\nTruth $p_T$ > 20 GeV, $|\eta| < 2.5$"
    plot_varVsVar = VarVsVarPlot(
        ylabel=r"Median $b$-jet "+label_map[y_variable]+" Response",
        xlabel=x_label,
        ylabel_ratio=["Signed RSD"],
        atlas_first_tag="Simulation Internal",
        atlas_second_tag = second_tag,
        # atlas_second_tag=r"$\sqrt{s} = 13$ TeV $t\overline{t}$ \n Truth $p_T$ > 20 GeV, $|\eta| < 2.5$",
        # atlas_second_tag=r"$Z'$ 250 < $p_T$ < 6000 GeV, $|\eta| < 2.5$",
        #atlas_tag_outside=True,
        logy=False,
        figsize=(6, 5),
        n_ratio_panels=1,
        ymin=0.9,
        ymax=1.2,
        ratio_method = "root_square_diff",
    )
    if ratio_mine:
        plot_varVsVar.add(response_median_reco)
    else:
        plot_varVsVar.add(response_median_reco, reference=True)
    # plot_varVsVar.add(response_median_pred)

    for i, name in enumerate(model_names):
        response_median_pred = VarVsVar(
            bin_centres,
            pred_medians_dict[name + model_df[i]],
            pred_medians_unc_dict[name + model_df[i]],
            x_var_widths=bin_widths,
            plot_y_std=False,
            label=f"{model_label[i]}"
        )
        plot_varVsVar.add(response_median_pred)

    if ratio_mine:
        y_ones = np.ones_like(bin_centres)
        y_zeros = np.zeros_like(bin_centres)  # zero uncertainty
        reference_flat = VarVsVar(
            bin_centres,
            y_ones,
            y_zeros,
            x_var_widths=bin_widths,
            plot_y_std=False,
        )
        reference_flat.linestyle = "dashed"
        reference_flat.colour = "black"
        plot_varVsVar.add(reference_flat, reference=True)

    plot_varVsVar.axis_top.axhline(1, c="k", ls=":", alpha=0.5)
    plot_varVsVar.draw()
    if plot_folder:
        plot_varVsVar.savefig(plot_folder+f"{y_variable}_response_median_vs_{x_variable}.png", transparent=False)
    else:
        plot_varVsVar.savefig(f"{y_variable}_response_median_vs_{x_variable}.png", transparent=False)

    plot_varVsVar = VarVsVarPlot(
        ylabel=r"Relative $b$-jet "+label_map[y_variable]+" Resolution",
        xlabel=x_label,
        ylabel_ratio=["Signed RSD"],
        atlas_first_tag="Simulation Internal",
        atlas_second_tag=second_tag,
        # atlas_second_tag=r"$\sqrt{s} = 13$ TeV $t\overline{t}$ \n Truth $p_T$ > 20 GeV, $|\eta| < 2.5$",
        # atlas_second_tag=r"$Z'$ 250 < $p_T$ < 6000 GeV, $|\eta| < 2.5$",
        #atlas_tag_outside=True,
        logy=False,
        figsize=(6, 5),
        n_ratio_panels=1,
        ratio_method = "root_square_diff",
    )
    plot_varVsVar.add(response_sigma_reco, reference=True)
    # plot_varVsVar.add(response_sigma_pred)


    for i, name in enumerate(model_names):
        response_sigma_pred = VarVsVar(
            bin_centres,
            pred_sigmas_dict[name + model_df[i]],
            pred_sigmas_unc_dict[name + model_df[i]],
            x_var_widths=bin_widths,
            plot_y_std=False,
            label=f"{model_label[i]}"
        )
        plot_varVsVar.add(response_sigma_pred)

    plot_varVsVar.draw()
    if plot_folder:
        plot_varVsVar.savefig(plot_folder+f"{y_variable}_response_resolution_vs_{x_variable}.png", transparent=False)
    else:
        plot_varVsVar.savefig(f"{y_variable}_response_resolution_vs_{x_variable}.png", transparent=False)






if __name__ == "__main__":
    filepath = "/share/lustre/avaitkus/salt/salt/logs/GN2XAux-fullData-50Epoch-flow_20231025-T100442/ckpts/epoch=037-val_loss=0.48245__test_pp_output_test.h5"

    from utils.data_utils import read_jets_from_test_file, find_tag_from_filepath, find_tag_from_phbb, standardise_jet_regression_labels

    jets_df = read_jets_from_test_file(filepath)
    
    tag1 = find_tag_from_filepath(filepath)
    tag2 = find_tag_from_phbb(jets_df)

    jets_df = standardise_jet_regression_labels(jets_df)

    print(list(jets_df.columns))

    plot_var_dist(jets_df, "mass", logy=True, bins=50, plot_filename="mass_dist.png")



