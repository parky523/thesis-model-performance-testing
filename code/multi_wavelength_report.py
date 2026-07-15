"""
multi_wavelength_report.py
Publication-quality multi-wavelength UV-Vis analysis figures for nitrate
in produced water samples.  Generates:

  Fig 1  – Multi-wavelength absorbance spectra (full range + UV zoom)
  Fig 2  – Single-wavelength calibration curves (220, 254, 275, 302 nm)
  Fig 3  – PLS-R calibration and LOO cross-validation
  Fig 4  – Results summary table (all samples)

Output folder: Pure_PW/Report_Figures/
CSV outputs  : Results_Table.csv, Calibration_Stats.csv
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import linregress
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR  = r"C:\Projects-2026\Project N\Pure_PW-20260121T112655Z-1-001\Pure_PW"
DATA_PATH = os.path.join(BASE_DIR, "Most Recent")
OUT_DIR   = os.path.join(BASE_DIR, "Report_Figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
FULL_START      = 190       # nm – start of full wavelength range to load
FULL_END        = 800       # nm – end of full wavelength range to load
PLS_WV_START    = 200       # nm – PLS model spectral region start
PLS_WV_END      = 400       # nm – PLS model spectral region end
PLS_N_LV        = 6         # latent variables (matching PLS-N.py)
DPI             = 300       # output figure resolution

# Key analytical wavelengths for nitrate UV-Vis
KEY_WL_VALUES  = [220, 254, 275, 302]
KEY_WL_LABELS  = [
    'NO$_3^-$ main (220 nm)',
    'UV$_{254}$ – DOC (254 nm)',
    'Organic corr. (275 nm)',
    'NO$_3^-$ secondary (302 nm)',
]
KEY_WL_SHORT   = ['220 nm', '254 nm', '275 nm', '302 nm']
KEY_WL_COLORS  = ['#1a6faf', '#2ca02c', '#ff7f0e', '#9467bd']
KEY_WL_MARKERS = ['v', 'D', 's', 'o']

# ── SAMPLE MAP (from PLS-N.py) ────────────────────────────────────────────────
target_map = {
    "PW_04": "27 ppm", "PW_1": "43 ppm", "PW_10": "44 ppm", "PW_11": "26 ppm",
    "PW_12": "34 ppm", "PW_13": "52 ppm", "PW_14": "50 ppm", "PW_15": "56 ppm",
    "PW_16": "7 ppm", "PW_17": "73 ppm", "PW_17A": "82 ppm", "PW_17B": "12 ppm",
    "PW_18": "42 ppm", "PW_19": "64 ppm", "PW_1B": "5 ppm", "PW_1C": "46 ppm",
    "PW_1D": "21 ppm", "PW_1E": "47 ppm", "PW_1F1": "27 ppm", "PW_2": "80 ppm",
    "PW_20": "46 ppm", "PW_20A": "27 ppm", "PW_20B": "33 ppm", "PW_20C": "28 ppm",
    "PW_21": "44 ppm", "PW_22A": "24 ppm", "PW_22B": "24 ppm", "PW_22C": "24 ppm",
    "PW_22D": "28 ppm", "PW_22E": "42 ppm", "PW_22F": "26 ppm", "PW_22G": "76 ppm",
    "PW_22H": "23 ppm", "PW_22I": "30 ppm", "PW_22J": "22 ppm", "PW_23": "42 ppm",
    "PW_23A": "46 ppm", "PW_24": "90 ppm", "PW_24A": "25 ppm", "PW_24B": "22 ppm",
    "PW_24C": "4 ppm", "PW_24D": "12 ppm", "PW_24E": "25 ppm", "PW_24F": "23 ppm",
    "PW_26": "32 ppm", "PW_27": "41 ppm", "PW_28": "45 ppm", "PW_28A": "34 ppm",
    "PW_28B": "52 ppm", "PW_28C": "32 ppm", "PW_29": "74 ppm", "PW_29.1": "26 ppm",
    "PW_29.2": "29 ppm", "PW_29.3": "25 ppm", "PW_2B": "11 ppm", "PW_2C": "28 ppm",
    "PW_2D": "4 ppm", "PW_2E": "20 ppm", "PW_2F": "25 ppm", "PW_2F0": "26 ppm",
    "PW_2F0.1": "43 ppm", "PW_2F0.2": "28 ppm", "PW_2F2": "40 ppm", "PW_3": "27 ppm",
    "PW_30": "46 ppm", "PW_30A": "36 ppm", "PW_31": "51 ppm", "PW_31A": "13 ppm",
    "PW_31B": "10 ppm", "PW_31C": "10 ppm", "PW_31D": "11 ppm", "PW_31E": "11 ppm",
    "PW_31F": "18 ppm", "PW_31G": "8 ppm", "PW_31H": "16 ppm", "PW_32": "48 ppm",
    "PW_32A": "46 ppm", "PW_36": "53 ppm", "PW_36A": "75 ppm", "PW_37": "44 ppm",
    "PW_37A": "54 ppm", "PW_38": "56 ppm", "PW_38A": "24 ppm", "PW_38B": "12 ppm",
    "PW_38C": "12 ppm", "PW_39": "47 ppm", "PW_39A": "90 ppm", "PW_39B": "38 ppm",
    "PW_39C": "35 ppm", "PW_39D": "23 ppm", "PW_39E": "23 ppm", "PW_39E.2": "19 ppm",
    "PW_39F": "11 ppm", "PW_39G": "9 ppm", "PW_39I": "12 ppm", "PW_39J": "54 ppm",
    "PW_3A": "82 ppm", "PW_3B": "5 ppm", "PW_40": "30 ppm", "PW_40A": "11 ppm",
    "PW_40B": "11 ppm", "PW_40C": "9 ppm", "PW_40D": "9 ppm", "PW_40E": "18 ppm",
    "PW_41": "68 ppm", "PW_42": "90 ppm", "PW_42A": "37 ppm", "PW_42B": "29 ppm",
    "PW_42C": "29 ppm", "PW_42D": "21 ppm", "PW_42E": "20 ppm", "PW_42F": "26 ppm",
    "PW_42I": "36 ppm", "PW_42J": "22 ppm", "PW_43": "73 ppm", "PW_44": "41 ppm",
    "PW_45": "24 ppm", "PW_45A": "72 ppm", "PW_45B": "10 ppm", "PW_45C": "6 ppm",
    "PW_45D": "8 ppm", "PW_45E": "9 ppm", "PW_45G": "11 ppm", "PW_46": "46 ppm",
    "PW_47": "22 ppm", "PW_47A": "32 ppm", "PW_49": "74 ppm", "PW_4F0": "35 ppm",
    "PW_4F2": "24 ppm", "PW_5": "54 ppm", "PW_50": "56 ppm", "PW_51": "77 ppm",
    "PW_51A": "136 ppm", "PW_51B": "32 ppm", "PW_51C": "90 ppm", "PW_51D": "18 ppm",
    "PW_51E": "20 ppm", "PW_52": "50 ppm", "PW_52A": "24 ppm", "PW_52B": "27 ppm",
    "PW_52C": "18 ppm", "PW_52D": "33 ppm", "PW_53": "23 ppm", "PW_53A": "117 ppm",
    "PW_53B": "58 ppm", "PW_54.2": "35 ppm", "PW_54.3": "38 ppm", "PW_6": "41 ppm",
    "PW_6A": "50 ppm", "PW_7": "44 ppm", "PW_7A": "60 ppm", "PW_7F0": "66 ppm",
    "PW_8": "53 ppm", "PW_9": "48 ppm", "PW_HS1": "42 ppm",
}

exclude_list = {'PW_29', 'PW_2', 'PW_2D', 'PW_16', 'PW_1B', 'PW_42E', 
    'PW_24C', 'PW_45', 'PW_51B', 'PW_31G', 'PW_2F0', 'PW_39I',
    'PW_45D', 'PW_45C', 'PW_31H', 'PW_42J', 'PW_51D', 
    'PW_31B', 'PW_1F1', 'PW_31A', 'PW_24D', 'PW_28A',
    'PW_3B', 'PW_45E', 'PW_31C'}
filtered_map = {k: v for k, v in target_map.items() if k not in exclude_list}

# ── LOAD SPECTRA ──────────────────────────────────────────────────────────────
print("Loading spectra from:", DATA_PATH)
spectra_list, y_list, id_list = [], [], []
wl_axis = None

for sid, target in filtered_map.items():
    fp = os.path.join(DATA_PATH, f"{sid}.spz")
    if not os.path.exists(fp):
        print(f"  [missing] {sid}.spz")
        continue
    try:
        df   = pd.read_csv(fp, sep='\t', skiprows=25, header=None, encoding='latin-1')
        wvs  = df.iloc[:, 0].values
        ints = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        mask = (wvs >= FULL_START) & (wvs <= FULL_END)
        spec = ints[mask]
        if len(spec) > 10:
            spectra_list.append(spec)
            y_list.append(float(re.findall(r"\d+", target)[0]))
            id_list.append(sid)
            if wl_axis is None:
                wl_axis = wvs[mask]
    except Exception as e:
        print(f"  [error] {sid}: {e}")

spectra    = np.array(spectra_list)
y_ppm      = np.array(y_list)
n          = len(id_list)
sort_idx   = np.argsort(y_ppm)
print(f"Loaded {n} samples | Range: {y_ppm.min():.0f}-{y_ppm.max():.0f} ppm\n")

# ── HELPER: nearest wavelength index ─────────────────────────────────────────
def widx(wl_target):
    return int(np.argmin(np.abs(wl_axis - wl_target)))

# Actual wavelengths used (nearest pixel to requested wavelength)
actual_wls = [wl_axis[widx(w)] for w in KEY_WL_VALUES]

# Absorbance at each key wavelength
abs_matrix = np.column_stack([spectra[:, widx(w)] for w in KEY_WL_VALUES])

# ── SNV + SAVITZKY-GOLAY for PLS ──────────────────────────────────────────────
pls_mask = (wl_axis >= PLS_WV_START) & (wl_axis <= PLS_WV_END)
X_raw    = spectra[:, pls_mask]
wl_pls   = wl_axis[pls_mask]

X_snv = (X_raw - X_raw.mean(axis=1, keepdims=True)) / X_raw.std(axis=1, keepdims=True)
X_pre = savgol_filter(X_snv, window_length=11, polyorder=2, deriv=1)

pls_model = PLSRegression(n_components=PLS_N_LV)
pls_model.fit(X_pre, y_ppm)
y_cal     = pls_model.predict(X_pre).ravel()

y_cv      = cross_val_predict(pls_model, X_pre, y_ppm, cv=LeaveOneOut()).ravel()

r2_cal   = r2_score(y_ppm, y_cal)
rmse_cal = np.sqrt(mean_squared_error(y_ppm, y_cal))
r2_cv    = r2_score(y_ppm, y_cv)
rmse_cv  = np.sqrt(mean_squared_error(y_ppm, y_cv))

print(f"PLS Calibration : R² = {r2_cal:.4f}  RMSE = {rmse_cal:.2f} ppm")
print(f"PLS LOO-CV      : R² = {r2_cv:.4f}  RMSE = {rmse_cv:.2f} ppm\n")

# ── GLOBAL STYLE ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'axes.linewidth':    1.0,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.top':         True,
    'ytick.right':       True,
    'legend.fontsize':   9,
    'legend.framealpha': 0.8,
})

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – Multi-Wavelength Absorbance Spectra
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Multi-wavelength spectra ...")

cmap = plt.cm.plasma
norm = mcolors.Normalize(vmin=y_ppm.min(), vmax=y_ppm.max())

fig1 = plt.figure(figsize=(15, 7))
gs1  = fig1.add_gridspec(1, 3, width_ratios=[7, 7, 0.4],
                         left=0.07, right=0.93, bottom=0.12, top=0.86,
                         wspace=0.05, hspace=0.05)
ax1a = fig1.add_subplot(gs1[0])   # full range
ax1b = fig1.add_subplot(gs1[1])   # UV zoom
ax1c = fig1.add_subplot(gs1[2])   # colorbar

fig1.suptitle(
    'UV–Vis Absorbance Spectra of Produced Water — Coloured by Nitrate Concentration',
    fontsize=13, fontweight='bold',
)

for ax, xlim, subtitle in [
    (ax1a, (FULL_START, FULL_END),        f'(a)  Full range: {FULL_START}–{FULL_END} nm'),
    (ax1b, (PLS_WV_START, PLS_WV_END),   f'(b)  Nitrate / UV region: {PLS_WV_START}–{PLS_WV_END} nm'),
]:
    for i in sort_idx:
        ax.plot(wl_axis, spectra[i], color=cmap(norm(y_ppm[i])),
                alpha=0.72, linewidth=0.85)

    # Key wavelength markers
    for wl_nm, label_s, col in zip(KEY_WL_VALUES, KEY_WL_SHORT, KEY_WL_COLORS):
        if xlim[0] <= wl_nm <= xlim[1]:
            ax.axvline(wl_nm, color=col, linestyle='--', linewidth=1.0, alpha=0.85)

    # Label at top inside axes
    ax.set_xlim(xlim)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for wl_nm, label_s, col in zip(KEY_WL_VALUES, KEY_WL_SHORT, KEY_WL_COLORS):
        if xlim[0] + 5 <= wl_nm <= xlim[1] - 5:
            ax.text(wl_nm + 1.5, 0.97, label_s, transform=trans,
                    fontsize=7.5, color=col, rotation=90,
                    va='top', ha='left', fontweight='bold')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_title(subtitle, loc='left', fontsize=11)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

ax1a.set_ylabel('Absorbance (AU)')
ax1b.set_yticklabels([])

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, cax=ax1c)
cb.set_label('[NO$_3^-$–N] (mg/L as ppm)', fontsize=11)

fig1.savefig(os.path.join(OUT_DIR, 'Fig1_Multiwavelength_Spectra.png'),
             dpi=DPI, bbox_inches='tight')
print("  Saved: Fig1_Multiwavelength_Spectra.png")
plt.close(fig1)

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 – Single-Wavelength Calibration Curves
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Single-wavelength calibration curves ...")

cal_stats_rows = []

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 11),
                            gridspec_kw={'hspace': 0.42, 'wspace': 0.32})
fig2.suptitle(
    'Single-Wavelength Calibration Curves\n'
    'Absorbance vs.  [NO$_3^-$–N] Concentration',
    fontsize=13, fontweight='bold',
)

panels = ['(a)', '(b)', '(c)', '(d)']

for ax, panel, wl_req, actual_wl, full_label, short_label, col, marker in zip(
    axes2.ravel(), panels,
    KEY_WL_VALUES, actual_wls,
    KEY_WL_LABELS, KEY_WL_SHORT,
    KEY_WL_COLORS, KEY_WL_MARKERS,
):
    y_abs = abs_matrix[:, KEY_WL_VALUES.index(wl_req)]

    # Linear regression
    slope, intercept, r_val, p_val, se = linregress(y_ppm, y_abs)
    r2_lin   = r_val ** 2
    residuals = y_abs - (slope * y_ppm + intercept)
    sigma_res = residuals.std(ddof=2)
    lod = 3  * sigma_res / abs(slope) if abs(slope) > 1e-9 else np.nan
    loq = 10 * sigma_res / abs(slope) if abs(slope) > 1e-9 else np.nan

    x_fit = np.linspace(y_ppm.min(), y_ppm.max(), 300)
    y_fit = slope * x_fit + intercept

    # 95 % confidence band
    n_fit   = len(y_ppm)
    x_mean  = y_ppm.mean()
    se_fit  = sigma_res * np.sqrt(1/n_fit + (x_fit - x_mean)**2 /
                                   np.sum((y_ppm - x_mean)**2))
    t_crit  = 2.0   # ~95 % for large n
    ci_up   = y_fit + t_crit * se_fit
    ci_dn   = y_fit - t_crit * se_fit

    ax.fill_between(x_fit, ci_dn, ci_up, alpha=0.15, color=col, label='95 % CI')
    ax.plot(x_fit, y_fit, color=col, linewidth=2.0, label='Linear fit', zorder=3)
    ax.scatter(y_ppm, y_abs, color=col, s=55, edgecolors='k',
               linewidths=0.5, zorder=4, marker=marker)

    # Annotate each point with sample ID (small)
    for xi, yi, sid in zip(y_ppm, y_abs, id_list):
        ax.annotate(sid.replace('PW_', ''), (xi, yi), fontsize=5.5,
                    xytext=(2.5, 2.5), textcoords='offset points', color='#444444')

    # Intercept sign formatting
    ic_sign = '+' if intercept >= 0 else ''
    stat_text = (
        f'$y = {slope:.5f}x {ic_sign}{intercept:.5f}$\n'
        f'$R^2 = {r2_lin:.4f}$\n'
        f'$RMSE = {residuals.std():.4f}$ AU\n'
        f'$LOD \\approx {lod:.1f}$ ppm\n'
        f'$LOQ \\approx {loq:.1f}$ ppm\n'
        f'$p = {p_val:.2e}$'
    )
    ax.text(0.04, 0.97, stat_text, transform=ax.transAxes, fontsize=8.5,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=col, alpha=0.9, linewidth=1.2))

    ax.set_xlabel('[NO$_3^-$–N] (ppm)')
    ax.set_ylabel(f'Absorbance at {actual_wl:.1f} nm (AU)')
    ax.set_title(f'{panel}  {full_label}', loc='left', fontsize=11)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    cal_stats_rows.append({
        'Panel':             panel,
        'Wavelength req (nm)': wl_req,
        'Wavelength used (nm)': f'{actual_wl:.2f}',
        'Assignment':        full_label,
        'Slope (AU/ppm)':    round(slope, 6),
        'Intercept (AU)':    round(intercept, 6),
        'R2':                round(r2_lin, 5),
        'RMSE (AU)':         round(residuals.std(), 5),
        'LOD (ppm)':         round(lod, 2) if not np.isnan(lod) else 'N/A',
        'LOQ (ppm)':         round(loq, 2) if not np.isnan(loq) else 'N/A',
        'p-value':           f'{p_val:.3e}',
        'n':                 n,
    })

fig2.savefig(os.path.join(OUT_DIR, 'Fig2_Calibration_Curves.png'),
             dpi=DPI, bbox_inches='tight')
print("  Saved: Fig2_Calibration_Curves.png")
plt.close(fig2)

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 – PLS-R Calibration & LOO Cross-Validation
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: PLS-R calibration ...")

ppm_min = y_ppm.min() - 3
ppm_max = y_ppm.max() + 3

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 6.5),
                            gridspec_kw={'wspace': 0.35})
fig3.suptitle(
    f'PLS-R Nitrate Prediction — SNV + 1st Derivative Savitzky–Golay (w=11, p=2)\n'
    f'{PLS_WV_START}–{PLS_WV_END} nm  |  {PLS_N_LV} Latent Variables  |  n = {n} samples',
    fontsize=12, fontweight='bold',
)

panels3   = ['(a)  Calibration', '(b)  Leave-One-Out Cross-Validation']
preds3    = [y_cal, y_cv]
colors3   = ['#1a6faf', '#c0392b']
r2s3      = [r2_cal, r2_cv]
rmses3    = [rmse_cal, rmse_cv]

for ax, panel, y_pred, col, r2, rmse in zip(
    axes3, panels3, preds3, colors3, r2s3, rmses3
):
    ax.plot([ppm_min, ppm_max], [ppm_min, ppm_max],
            'k--', linewidth=1.2, alpha=0.55, label='1:1 line', zorder=1)

    sc = ax.scatter(y_ppm, y_pred, c=col, s=70, edgecolors='k',
                    linewidths=0.5, alpha=0.88, zorder=3, label='Samples')

    for xi, yi, sid in zip(y_ppm, y_pred, id_list):
        ax.annotate(sid.replace('PW_', ''), (xi, yi), fontsize=6,
                    xytext=(3, 3), textcoords='offset points', color='#333333')

    stat_text = (
        f'$R^2 = {r2:.4f}$\n'
        f'$RMSE = {rmse:.2f}$ ppm\n'
        f'$n = {n}$\n'
        f'LVs $= {PLS_N_LV}$'
    )
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=10,
            va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#eef6fc',
                                edgecolor=col, alpha=0.9, linewidth=1.2))

    ax.set_xlabel('Actual [NO$_3^-$–N] (ppm)')
    ax.set_ylabel('Predicted [NO$_3^-$–N] (ppm)')
    ax.set_title(panel, loc='left', fontsize=11)
    ax.set_xlim(ppm_min, ppm_max)
    ax.set_ylim(ppm_min, ppm_max)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.legend(loc='lower right', fontsize=9)

fig3.savefig(os.path.join(OUT_DIR, 'Fig3_PLS_Calibration.png'),
             dpi=DPI, bbox_inches='tight')
print("  Saved: Fig3_PLS_Calibration.png")
plt.close(fig3)

# ═════════════════════════════════════════════════════════════════════════════
#  RESULTS TABLE – CSV + Figure
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Results summary table ...")

rows = []
for i, sid in enumerate(id_list):
    row = {
        'Sample ID':        sid,
        'Actual (ppm)':     int(y_ppm[i]),
        'Abs@220nm (AU)':   round(abs_matrix[i, 0], 4),
        'Abs@254nm (AU)':   round(abs_matrix[i, 1], 4),
        'Abs@275nm (AU)':   round(abs_matrix[i, 2], 4),
        'Abs@302nm (AU)':   round(abs_matrix[i, 3], 4),
        'PLS Cal. (ppm)':   round(y_cal[i], 1),
        'PLS LOO-CV (ppm)': round(y_cv[i],  1),
        'CV Residual (ppm)':round(y_cv[i] - y_ppm[i], 1),
    }
    rows.append(row)

results_df = pd.DataFrame(rows).sort_values('Actual (ppm)').reset_index(drop=True)
results_df.to_csv(os.path.join(OUT_DIR, 'Results_Table.csv'), index=False)
print("  Saved: Results_Table.csv")

# Calibration statistics table
cal_df = pd.DataFrame(cal_stats_rows)
cal_df.to_csv(os.path.join(OUT_DIR, 'Calibration_Stats.csv'), index=False)
print("  Saved: Calibration_Stats.csv")

# ── Figure 4a: Results table ──────────────────────────────────────────────────
col_labels = list(results_df.columns)
cell_text  = [
    [str(v) for v in row] for row in results_df.itertuples(index=False)
]
n_rows_t   = len(results_df)

fig4_h = max(9, n_rows_t * 0.31 + 3.0)
fig4, ax4 = plt.subplots(figsize=(20, fig4_h))
ax4.axis('off')
fig4.suptitle(
    'Table 1.  Summary Results: UV–Vis Absorbance and PLS-R Nitrate Prediction\n'
    f'Produced Water Samples | n = {n} | '
    f'PLS LOO-CV: R² = {r2_cv:.4f}, RMSE = {rmse_cv:.2f} ppm',
    fontsize=12, fontweight='bold', y=0.99,
)

tbl = ax4.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc='upper center',
    cellLoc='center',
    bbox=[0, 0, 1, 0.98],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.auto_set_column_width(col=list(range(len(col_labels))))

# Header style
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=8.5)
    cell.set_height(0.045)

# Zebra rows + highlight large CV residuals
for row_i in range(1, n_rows_t + 1):
    for col_i, col_name in enumerate(col_labels):
        cell = tbl[row_i, col_i]
        bg = '#f7f9fc' if row_i % 2 == 0 else 'white'
        cell.set_facecolor(bg)
        cell.set_height(0.033)
        if col_name == 'CV Residual (ppm)':
            try:
                err = float(cell.get_text().get_text())
                if abs(err) > 5:
                    cell.set_facecolor('#ffe4e1')
                elif abs(err) <= 2:
                    cell.set_facecolor('#e8f8e8')
            except ValueError:
                pass

fig4.tight_layout(rect=[0, 0, 1, 0.97])
fig4.savefig(os.path.join(OUT_DIR, 'Fig4_Results_Table.png'),
             dpi=DPI, bbox_inches='tight')
print("  Saved: Fig4_Results_Table.png")
plt.close(fig4)

# ── Figure 4b: Calibration statistics table ───────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(17, 4.5))
ax5.axis('off')
fig5.suptitle(
    'Table 2.  Single-Wavelength Calibration Statistics — Absorbance vs. [NO$_3^-$–N]',
    fontsize=12, fontweight='bold', y=1.02,
)

cal_cols = list(cal_df.columns)
cal_cells = [[str(v) for v in row] for row in cal_df.itertuples(index=False)]

tbl2 = ax5.table(
    cellText=cal_cells,
    colLabels=cal_cols,
    loc='center',
    cellLoc='center',
    bbox=[0, 0, 1, 1.0],
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(8.5)
tbl2.auto_set_column_width(col=list(range(len(cal_cols))))

for j in range(len(cal_cols)):
    cell = tbl2[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=8.5)
    cell.set_height(0.22)

for row_i in range(1, len(cal_df) + 1):
    for col_i in range(len(cal_cols)):
        cell = tbl2[row_i, col_i]
        cell.set_facecolor('#f0f4f8' if row_i % 2 == 0 else 'white')
        cell.set_height(0.18)

fig5.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, 'Fig5_Calibration_Stats_Table.png'),
             dpi=DPI, bbox_inches='tight')
print("  Saved: Fig5_Calibration_Stats_Table.png")
plt.close(fig5)

# ── Console summary ───────────────────────────────────────────────────────────
SEP = "=" * 70
print("\n" + SEP)
print("  PLS-R MODEL SUMMARY")
print(SEP)
print(f"  Spectral range   : {PLS_WV_START}-{PLS_WV_END} nm")
print(f"  Preprocessing    : SNV + Savitzky-Golay (1st deriv, w=11, p=2)")
print(f"  Latent variables : {PLS_N_LV}")
print(f"  Training (Cal)   : R2 = {r2_cal:.4f}  |  RMSE = {rmse_cal:.2f} ppm")
print(f"  LOO Cross-Val    : R2 = {r2_cv:.4f}  |  RMSE = {rmse_cv:.2f} ppm")
print(f"  Samples (n)      : {n}")
print("-" * 70)
print("\n  SINGLE-WAVELENGTH CALIBRATION STATISTICS")
print("-" * 70)
header_row = (f"  {'WL (nm)':<10} {'Assignment':<30} {'Slope':>12} "
              f"{'R2':>8} {'LOD (ppm)':>12} {'LOQ (ppm)':>12}")
print(header_row)
print("  " + "-" * 68)
for r in cal_stats_rows:
    print(f"  {r['Wavelength used (nm)']:<10} {r['Assignment']:<30} "
          f"{r['Slope (AU/ppm)']:>12.5f} {r['R2']:>8.4f} "
          f"{str(r['LOD (ppm)']):>12} {str(r['LOQ (ppm)']):>12}")
print("\n" + SEP)
print(f"  All outputs saved to:\n  {OUT_DIR}")
print(SEP)
