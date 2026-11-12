"""Orient saved width-10 iVAE weights to the published DDE shallow display."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
from paths import configure, inputs, output
parser=configure(argparse.ArgumentParser(description='Render saved width-10 iVAE and DDE matrices; no training.'))
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args();settings=inputs(args.config)
O=output(args.output_dir);I=O/'images';I.mkdir()
ref=settings['ivae'];weights=ref/'iVAE_plot_matrices_10_1.npz'
frame = pd.read_csv(ref / "all_standardized_loadings.csv")
D = frame[[f"DDE_{k}" for k in range(1, 11)]].to_numpy()
G = frame[[f"Gaussian_{k}" for k in range(1, 33)]].to_numpy()
saved = np.load(weights)
assert D.shape == saved["encoder"].shape == saved["decoder"].shape == (98, 10)
meta = pd.read_csv(ref / "item_metadata.csv")
assert frame["item_id"].tolist() == meta["item_id"].tolist()
group = np.argmax(abs(D), axis=1)
order = np.lexsort((np.arange(98), -abs(D[np.arange(98), group]), group))
assert np.array_equal(np.sort(order), np.arange(98))
breaks = np.flatnonzero(np.diff(group[order])) + 0.5
centers = [np.flatnonzero(group[order] == k).mean() for k in range(10)]
labels = [f"D{k+1}" for k in range(10)]
rowmap = frame.iloc[order, :4].copy()
rowmap.insert(0, "display_row_1based", np.arange(1, 99))
rowmap.insert(1, "DDE_group", group[order] + 1)
rowmap.to_csv(O / "ROW_ORDER.csv", index=False)

# Permute only. Hidden-unit sign flips are not assumed to be symmetries
# of the fitted leaky-ReLU networks.
matches = {}
panels = [(D, "DDE copula\n10 first-layer features", "B5_B1_DDE_factor_order",
           np.arange(1, 11), "DDE feature", "Standardized structural loading",
           max(abs(D).max(), abs(G).max()))]
audit_rows = []
for key, title, stem in [
    ("encoder", "iVAE encoder\n10 hidden units", "B5_enc_vae_factor_order"),
    ("decoder", "iVAE decoder\n10 hidden units", "B5_dec_vae_factor_order"),
]:
    A = saved[key]
    cosine = D.T @ A / (np.linalg.norm(D, axis=0)[:, None] * np.linalg.norm(A, axis=0)[None, :])
    ii, jj = linear_sum_assignment(-cosine)
    assert np.array_equal(ii, np.arange(10))
    assert np.array_equal(np.sort(jj), np.arange(10))
    matched = A[:, jj]
    assert np.array_equal(matched[:, np.argsort(jj)], A)
    matches[key] = jj
    for k, unit in enumerate(jj):
        audit_rows.append({"network": key, "DDE_column": k+1,
                           "original_hidden_unit": int(unit+1),
                           "cosine_similarity": float(cosine[k, unit]),
                           "sign_multiplier": 1})
    panels.append((matched, title, stem, jj+1,
                   "Original hidden unit (matched order)", "Network weight",
                   abs(A).max()))
pd.DataFrame(audit_rows).to_csv(O / "COLUMN_MATCHES.csv", index=False)
np.savez(O / "DISPLAY_MATRICES.npz",
         dde=D[order], encoder=saved["encoder"][order][:, matches["encoder"]],
         decoder=saved["decoder"][order][:, matches["decoder"]],
         row_order_0based=order, group_1based=group[order]+1,
         encoder_units_1based=matches["encoder"]+1,
         decoder_units_1based=matches["decoder"]+1)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})

def panel(ax, data, title, ticks, xlabel, clim):
    im = ax.imshow(data[order], aspect="auto", cmap="RdBu_r",
                   vmin=-clim, vmax=clim, interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(10), ticks, rotation=90, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_yticks(centers, labels)
    ax.set_ylabel("Items grouped by strongest DDE loading")
    for boundary in breaks:
        ax.axhline(boundary, color="#444444", lw=.6)
    return im

def save(fig, stem):
    for ext in ["pdf", "png"]:
        fig.savefig(I / f"{stem}.{ext}", dpi=260, bbox_inches="tight")
    plt.close(fig)

for data, title, stem, ticks, xlabel, colorlabel, clim in panels:
    fig, ax = plt.subplots(figsize=(4.8, 8.2), layout="constrained")
    im = panel(ax, data, title, ticks, xlabel, clim)
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.045, pad=.03,
                 label=colorlabel)
    save(fig, stem)

fig, axs = plt.subplots(1, 3, figsize=(12.4, 8.2), layout="constrained")
for ax, (data, title, stem, ticks, xlabel, colorlabel, clim) in zip(axs, panels):
    im = panel(ax, data, title, ticks, xlabel, clim)
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.045, pad=.03,
                 label=colorlabel)
save(fig, "iVAE_comparison_DDE_oriented")

print("Rendered the four saved-matrix iVAE/DDE panels and permutation audits.")
