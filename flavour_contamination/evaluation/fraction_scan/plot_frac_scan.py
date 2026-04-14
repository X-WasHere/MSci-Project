"""Produce parameterized efficiency fraction scans for f_c and f_tau (GN3 7-class)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import glob

from ftag.utils import calculate_efficiency
from ftag.cuts import Cuts
from ftag.hdf5.h5reader import H5Reader

from puma import Line2D, Line2DPlot
from puma.utils import logger, get_good_colours

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def get_details(df, label, seven_class=True):
    '''Defines a boolean array to select the different flavour classes'''
    is_light = df[label] == 0
    is_c = df[label] == 4
    is_tau = df[label] == 15
    
    results = {'is_light': is_light, 'is_c': is_c, 'is_tau': is_tau}

    if seven_class:
        results['is_trueb'] = (df[label] == 5) & (df['is_contaminated_bjet'] == False)
        results['is_contaminatedb'] = (df[label] == 5) & (df['is_contaminated_bjet'] == True)
    else:
        results['is_b'] = (df[label] == 5)

    return results

def calc_discriminant(df, name, fc, ftau, fcontam=0.0, split_contaminated_combined=True):
    """Calculates the log-likelihood ratio discriminant for the 7-class model."""
    if split_contaminated_combined:
        uscore = df[f'{name}_ps'] + df[f'{name}_pud'] + df[f'{name}_pgluon'] 
        nom = df[f'{name}_ptrueb'] + df[f'{name}_pcontamintedb'] + 1e-10
        flight = max(1 - fc - ftau - fcontam, 0.0) 
        denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + \
                (fcontam * df[f'{name}_pcontamintedb']) + (flight * uscore) + 1e-10
        return np.log(nom / denom)

def get_optimal_fraction_value(fraction_scan: np.ndarray, fraction_space: np.ndarray, rej: bool = False) -> tuple[int, float]:
    """Find the optimal fraction value (closest to origin for efficiency)."""
    xs, ys = np.copy(fraction_scan[:, 0]), np.copy(fraction_scan[:, 1])
    xs /= np.max(xs)
    ys /= np.max(ys)
    opt_idx = np.argmax(xs**2 + ys**2) if rej else np.argmin(xs**2 + ys**2)
    return opt_idx, fraction_space[opt_idx]

def load_df(file_pattern, model_name, num_jets=None, batch_size=500_000, seven_class=True):
    file_paths = glob.glob(file_pattern)
    cuts = Cuts.from_list([("pt", ">", 250000), ("pt", "<", 6000000), ("eta", "<", 2.5), ("eta", ">", -2.5)])
    jet_vars = [
        "pt", "eta", "eventNumber", "HadronGhostTruthLabelID", "flavour_label", "is_contaminated_bjet",
        f"{model_name}_ptrueb", f"{model_name}_pcontamintedb", f"{model_name}_pc", 
        f"{model_name}_ptau", f"{model_name}_ps", f"{model_name}_pud", f"{model_name}_pgluon"
    ]
    reader = H5Reader(fname=file_paths, jets_name="jets", batch_size=batch_size, do_remove_inf=True, shuffle=True)
    data = reader.load(variables={"jets": jet_vars}, cuts=cuts, num_jets=num_jets)
    return pd.DataFrame(data["jets"])


# ==========================================
# 2. LOAD DATA
# ==========================================

fname_zprime = '/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_benchmark/ckpts/epoch=002-val_loss=0.96618__test_zprime.h5'
fname_qcd = '/home/xzcapfed/MSci/flavour_contamination/logs/flavour_contamination_benchmark/ckpts/epoch=002-val_loss=0.96618__test_qcd.h5'
model_name = "GN3_calo_benchmark"

logger.info("Loading 7-Class datasets...")
df = pd.concat([load_df(fname_zprime, model_name), load_df(fname_qcd, model_name)])

flavours = get_details(df, "HadronGhostTruthLabelID", seven_class=True)
is_b = flavours["is_trueb"] | flavours["is_contaminatedb"]
is_c = flavours["is_c"]
is_tau = flavours["is_tau"]
is_light = flavours["is_light"]

# GLOBAL SCAN SETTINGS
SIG_EFF = 0.70  # Changed to 70% WP
FCONTAM_FIXED = 0.05
SCALING_FACTOR = 1.3 # Adjust this to match your subgroup's "scaled auto" definition

# ==========================================
# 3. f_c SCAN
# ==========================================
logger.info("Running f_c scan...")
fc_values = np.linspace(0.01, 0.40, 40)
FTAU_FIXED = 0.10

def calc_effs_fc(fc_value):
    disc = calc_discriminant(df, model_name, fc=fc_value, ftau=FTAU_FIXED, fcontam=FCONTAM_FIXED)
    return [fc_value, calculate_efficiency(disc[is_b].values, disc[is_light].values, SIG_EFF), 
            calculate_efficiency(disc[is_b].values, disc[is_c].values, SIG_EFF)]

eff_results_fc = np.array(list(map(calc_effs_fc, fc_values)))
x_fc, y_fc = eff_results_fc[:, 2], eff_results_fc[:, 1]

# Find Markers
opt_idx_fc, fc_auto = get_optimal_fraction_value(np.column_stack((x_fc, y_fc)), fc_values, rej=False)
fc_scaled = min(fc_auto * SCALING_FACTOR, np.max(fc_values))
print(f"RESULTS: Auto f_c is {fc_auto:.3f} | Scaled Auto f_c is {fc_scaled:.3f}")

idx_fixed = (np.abs(fc_values - 0.07)).argmin()
idx_scaled = (np.abs(fc_values - fc_scaled)).argmin()

# Plot f_c
plot_fc = Line2DPlot(figsize=(6, 5), atlas_second_tag=f"$\sqrt{{s}} = 13.6$ TeV, $Z'$ & QCD\n{int(SIG_EFF*100)}% $b$-jet WP")
plot_fc.add(Line2D(x_values=x_fc, y_values=y_fc, label="$f_c$ contour", colour="k", linestyle="-"))

# Add markers (Fixed, Auto, Scaled Auto)
plot_fc.add(Line2D(x_values=x_fc[idx_fixed], y_values=y_fc[idx_fixed], colour="r", marker="x", label=f"$f_c=0.07$", markersize=12, markeredgewidth=2), is_marker=True)
plot_fc.add(Line2D(x_values=x_fc[opt_idx_fc], y_values=y_fc[opt_idx_fc], colour="g", marker="x", label="auto", markersize=12, markeredgewidth=2), is_marker=True)
plot_fc.add(Line2D(x_values=x_fc[idx_scaled], y_values=y_fc[idx_scaled], colour="b", marker="x", label="scaled auto", markersize=12, markeredgewidth=2), is_marker=True)

plot_fc.ylabel, plot_fc.xlabel = "Light-flavour jet efficiency", "$c$-jet efficiency"
plot_fc.draw()
plot_fc.savefig("/home/xzcapfed/MSci/flavour_contamination/evaluation/fraction_scan/efficiencyScanPlot_fc_7class_zprime.png", transparent=False)

# ==========================================
# 4. f_tau SCAN
# ==========================================
logger.info(f"Running f_tau scan (using f_c auto = {fc_auto:.3f})...")
ftau_values = np.linspace(0.01, 0.30, 40)

def calc_effs_ftau(ftau_value):
    disc = calc_discriminant(df, model_name, fc=fc_scaled, ftau=ftau_value, fcontam=FCONTAM_FIXED)
    return [ftau_value, calculate_efficiency(disc[is_b].values, disc[is_light].values, SIG_EFF), 
            calculate_efficiency(disc[is_b].values, disc[is_tau].values, SIG_EFF)]

eff_results_ftau = np.array(list(map(calc_effs_ftau, ftau_values)))
x_ftau, y_ftau = eff_results_ftau[:, 2], eff_results_ftau[:, 1]

# Find Markers
opt_idx_ftau, ftau_auto = get_optimal_fraction_value(np.column_stack((x_ftau, y_ftau)), ftau_values, rej=False)
ftau_scaled = min(ftau_auto * SCALING_FACTOR, np.max(ftau_values))
print(f"RESULTS: Auto f_tau is {ftau_auto:.3f} | Scaled Auto f_tau is {ftau_scaled:.3f}")

idx_fixed_tau = (np.abs(ftau_values - 0.02)).argmin() # Common baseline for tau
idx_scaled_tau = (np.abs(ftau_values - ftau_scaled)).argmin()

# Plot f_tau
plot_ftau = Line2DPlot(figsize=(6, 5), atlas_second_tag=f"$\sqrt{{s}} = 13.6$ TeV, $Z'$ & QCD\n{int(SIG_EFF*100)}% $b$-jet WP\nFixed $f_c={fc_auto:.3f}$")
plot_ftau.add(Line2D(x_values=x_ftau, y_values=y_ftau, label="$f_\\tau$ contour", colour="k", linestyle="-"))

# Add markers (Fixed, Auto, Scaled Auto)
plot_ftau.add(Line2D(x_values=x_ftau[idx_fixed_tau], y_values=y_ftau[idx_fixed_tau], colour="r", marker="x", label=f"$f_\\tau=0.02$", markersize=12, markeredgewidth=2), is_marker=True)
plot_ftau.add(Line2D(x_values=x_ftau[opt_idx_ftau], y_values=y_ftau[opt_idx_ftau], colour="g", marker="x", label="auto", markersize=12, markeredgewidth=2), is_marker=True)
plot_ftau.add(Line2D(x_values=x_ftau[idx_scaled_tau], y_values=y_ftau[idx_scaled_tau], colour="b", marker="x", label="scaled auto", markersize=12, markeredgewidth=2), is_marker=True)

plot_ftau.ylabel, plot_ftau.xlabel = "Light-flavour jet efficiency", "$\\tau$-jet efficiency"
plot_ftau.draw()
plot_ftau.savefig("/home/xzcapfed/MSci/flavour_contamination/evaluation/fraction_scan/efficiencyScanPlot_ftau_7class_zprime.png", transparent=False)

logger.info("Both scans complete and saved!")