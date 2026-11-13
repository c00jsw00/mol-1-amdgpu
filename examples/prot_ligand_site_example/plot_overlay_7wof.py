#!/usr/bin/env python3
"""Three-structure overlay figure for the 7WOF comparison example.
Reads overlay_7wof_data.npy (from compare_with_7wof.py) and writes
overlay_7wof.png next to this script. Needs numpy + matplotlib."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa

DIR = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(DIR, "overlay_7wof_data.npy"), allow_pickle=True).item()

STYLE = {
    "exp": dict(color="0.3", lw=2.0, ls="-", label="7WOF experimental"),
    "mol-1": dict(color="tab:blue", lw=1.4, ls="--", label="mol-1 (base)"),
    "mol-1-constraint": dict(color="tab:red", lw=1.4, ls="-.", label="mol-1-constraint (SITE)"),
}

def ca_pts(key):
    if key == "exp":
        return d["exp"]["ca"]
    p = d[key]
    return np.array([p["ca"][i] for i in sorted(p["ca"])])

def lig_pts(key):
    return d["exp"]["lig"] if key == "exp" else d[key]["lig"]

def sticks(ax, pts, color, lw=2.4, alpha=1.0):
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if 0.1 < np.linalg.norm(pts[i] - pts[j]) < 1.85:
                ax.plot(*zip(pts[i], pts[j]), color=color, lw=lw, alpha=alpha, solid_capstyle="round")

fig = plt.figure(figsize=(14, 6.8))

ax1 = fig.add_subplot(121, projection="3d")
for key in ("exp", "mol-1", "mol-1-constraint"):
    s = STYLE[key]
    p = ca_pts(key)
    ax1.plot(p[:, 0], p[:, 1], p[:, 2], color=s["color"], lw=s["lw"], ls=s["ls"], alpha=0.85)
    seg = d["exp"]["site"] if key == "exp" else np.array(d[key]["site"])
    ax1.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=s["color"], lw=5, alpha=0.9)
    sticks(ax1, lig_pts(key), s["color"], lw=2.0, alpha=0.9)
c = ca_pts("exp").mean(0)
for meth, i in (("set_xlim", 0), ("set_ylim", 1), ("set_zlim", 2)):
    getattr(ax1, meth)(c[i] - 42, c[i] + 42)
ax1.set_title("whole protein: CA traces + GC376 poses\n(thick = H41, M49, N142-G143-S144-C145)", fontsize=11)
ax1.view_init(elev=10, azim=-60)

cen = np.array(d["exp"]["lig"]).mean(0)
ax2 = fig.add_subplot(122, projection="3d")
R = 12.0
for key in ("exp", "mol-1", "mol-1-constraint"):
    p = ca_pts(key)
    dist = np.linalg.norm(p - cen, axis=1)
    s = STYLE[key]
    for k in range(len(p) - 1):
        if dist[k] < R + 3 and dist[k + 1] < R + 3:
            ax2.plot(*zip(p[k], p[k + 1]), color=s["color"], lw=1.8, ls=s["ls"], alpha=0.8)
    sticks(ax2, lig_pts(key), s["color"], lw=3.4)
for meth, i in (("set_xlim", 0), ("set_ylim", 1), ("set_zlim", 2)):
    getattr(ax2, meth)(cen[i] - R, cen[i] + R)
ax2.set_title("binding pocket zoom (+/-12 A)\nligand RMSD vs 7WOF: mol-1 1.28 A, constraint 0.66 A", fontsize=11)
ax2.view_init(elev=15, azim=-75)

handles = [Line2D([0], [0], color=STYLE[k]["color"], lw=2.5, ls=STYLE[k]["ls"], label=STYLE[k]["label"])
           for k in ("exp", "mol-1", "mol-1-constraint")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=True)
fig.suptitle("SARS-CoV-2 Mpro + GC376 - experimental 7WOF vs Mol-1 predictions (AMD ROCm)\n"
             "protein CA RMSD: mol-1 0.71 A | mol-1-constraint 0.69 A", fontsize=13)
fig.subplots_adjust(top=0.82, bottom=0.12)
fig.savefig(os.path.join(DIR, "overlay_7wof.png"), dpi=150, bbox_inches="tight")
print("wrote overlay_7wof.png")
