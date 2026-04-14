import h5py
import numpy as np
from ftag import Flavours
from ftag.utils import get_discriminant
from ftag.cuts import Cuts
from puma import VarVsEff, VarVsEffPlot
from puma.utils import logger

def calc_discriminant(df, name, fc, ftau):
    uscore = df[f'{name}_ps'] + df[f'{name}_pud'] + df[f'{name}_pg'] # prob of light jet is the sum of these components
    nom = df[f'{name}_pb'] + 1e-10
    denom = (fc * df[f'{name}_pc']) + (ftau * df[f'{name}_ptau']) + ((1-fc-ftau) * uscore) + 1e-10
    disc = nom/denom  

    return np.log(disc)

fname='/home/xzcapfed/MSci/GN3_calo/logs/new-data-benchmark/ckpts/epoch=004-val_loss=0.97334__test_ttbar.h5'
fname_benchmark='/home/xzcapfed/MSci/GN3_calo/logs/benchmark/ckpts/epoch=003-val_loss=0.99337__test_ttbar.h5'
# fname='/home/xzcapfed/MSci/GN3_calo/logs/new-data-benchmark/ckpts/epoch=004-val_loss=0.97334__test_zprime.h5'
# fname_benchmark='/home/xzcapfed/MSci/GN3_calo/logs/benchmark/ckpts/epoch=003-val_loss=0.99337__test_zprime.h5'

# 2. Define your kinematic cuts
cuts = Cuts.from_list([
    ("pt", ">", 20000),
    ("pt", "<", 250000),
    # ("pt", ">", 250000),
    # ("pt", "<", 6000000),
    ("eta", "<", 2.5),
    ("eta", ">", -2.5),
])


logger.info("Loading and filtering jets from H5 file")
with h5py.File(fname, 'r') as f:
    jets_calo = f["jets"][:]
with h5py.File(fname_benchmark, 'r') as f:
    jets_benchmark = f["jets"][:]

# cuts(jets).idx returns bool mask
jets_calo = jets_calo[cuts(jets_calo).idx]
jets_benchmark = jets_benchmark[cuts(jets_benchmark).idx]

pt_calo = jets_calo["pt"] / 1000
pt_benchmark = jets_benchmark["pt"] / 1000


logger.info("Calculating discriminants")
disc_calo = calc_discriminant(jets_calo, "GN3_calo", fc=0.2, ftau=0.1)
disc_benchmark = calc_discriminant(jets_benchmark, "GN3_calo_benchmark", fc=0.2, ftau=0.1)

# selecting jets
is_b_calo = Flavours["bjets"].cuts(jets_calo).idx
is_c_calo = Flavours["cjets"].cuts(jets_calo).idx
is_light_calo = Flavours["ujets"].cuts(jets_calo).idx
is_b_benchmark = Flavours["bjets"].cuts(jets_benchmark).idx
is_c_benchmark = Flavours["cjets"].cuts(jets_benchmark).idx
is_light_benchmark = Flavours["ujets"].cuts(jets_benchmark).idx


if 'ttbar' in fname:
    second_tag = "$\\sqrt{s} = 13.6$ TeV, $t\\overline{t}$ events\n20 GeV < $p_T$ < 250 GeV"
    bins = np.linspace(20, 250, 20)
    suffix = "ttbar"
else:
    second_tag = "$\\sqrt{s}=13.6$ TeV, $Z'$ events\n$250$ GeV $< p_{T} <6$ TeV"
    bins = [250, 260, 270, 300, 350, 400, 600, 700, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    suffix = "zprime"

fixed_lightjet_rejection = 1000 
fixed_cjet_rejection = 200 

# Light jet efficiency
logger.info("Setting up light-jet VarVsEff objects")
plot_path="/home/xzcapfed/MSci/GN3_calo/plots/eff_vs_pt/fixed_rej/"
eff_benchmark = VarVsEff(
    x_var_sig=pt_benchmark[is_b_benchmark],
    disc_sig=disc_benchmark[is_b_benchmark],
    x_var_bkg=pt_benchmark[is_light_benchmark],
    disc_bkg=disc_benchmark[is_light_benchmark],
    bins=bins,
    working_point=None,
    fixed_bkg_rej=fixed_lightjet_rejection,
    flat_per_bin=False,
    label="GN3 benchmark",
)
eff_benchmark.colour = "tab:blue"

eff_calo = VarVsEff(
    x_var_sig=pt_calo[is_b_calo],
    disc_sig=disc_calo[is_b_calo],
    x_var_bkg=pt_calo[is_light_calo],
    disc_bkg=disc_calo[is_light_calo],
    bins=bins,
    working_point=None,
    fixed_bkg_rej=fixed_lightjet_rejection,
    flat_per_bin=False,
    label="GN3 calo",
)
eff_calo.colour = "firebrick"

logger.info("Drawing plot")
plot_sig_eff = VarVsEffPlot(
    mode="sig_eff",
    ylabel="$b$-jet efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag=f"{second_tag}\nFixed light-jet rejection = {fixed_lightjet_rejection}",
    n_ratio_panels=1,
    figsize=(5.5, 4.5)
)

plot_sig_eff.add(eff_benchmark, reference=True)
plot_sig_eff.add(eff_calo)

plot_sig_eff.draw()
plot_sig_eff.savefig(f"{plot_path}beff_vs_pt_fixed_urej_{suffix}.png", transparent=False)


# c jet efficiency
logger.info("Setting up c-jet VarVsEff objects")
eff_benchmark = VarVsEff(
    x_var_sig=pt_benchmark[is_b_benchmark],
    disc_sig=disc_benchmark[is_b_benchmark],
    x_var_bkg=pt_benchmark[is_c_benchmark],
    disc_bkg=disc_benchmark[is_c_benchmark],
    bins=bins,
    working_point=None,
    fixed_bkg_rej=fixed_cjet_rejection,
    flat_per_bin=False,
    label="GN3 benchmark",
)
eff_benchmark.colour = "tab:blue"

eff_calo = VarVsEff(
    x_var_sig=pt_calo[is_b_calo],
    disc_sig=disc_calo[is_b_calo],
    x_var_bkg=pt_calo[is_c_calo],
    disc_bkg=disc_calo[is_c_calo],
    bins=bins,
    working_point=None,
    fixed_bkg_rej=fixed_cjet_rejection,
    flat_per_bin=False,
    label="GN3 calo",
)
eff_calo.colour = "firebrick"

logger.info("Drawing plot")
plot_sig_eff = VarVsEffPlot(
    mode="sig_eff",
    ylabel="$b$-jet efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag=f"{second_tag}\nFixed $c$-jet rejection = {fixed_cjet_rejection}",
    n_ratio_panels=1,
    figsize=(5.5, 4.5)
)

plot_sig_eff.add(eff_benchmark, reference=True)
plot_sig_eff.add(eff_calo)

plot_sig_eff.draw()
plot_sig_eff.savefig(f"{plot_path}beff_vs_pt_fixed_crej_{suffix}.png", transparent=False)