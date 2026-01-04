#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from typing import List, Optional

# Matplotlib writes cache files (including TeX-related caches when usetex=True).
# Ensure a writable cache directory even in sandboxed / restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = os.path.join(tempfile.gettempdir(), "chemotaxis-sim-mplconfig")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_config_dir

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter


USE_TEX = os.environ.get("CHEMOTAXIS_SIM_USETEX", "yes").strip().lower() not in (
    "0",
    "no",
    "false",
)
rc("text", usetex=USE_TEX)


def create_six_frame_summary(
    x_values: np.ndarray,
    t_values: np.ndarray,
    u_num: np.ndarray,
    uStar: float,
    chi0: float,
    chi_star: float,
    file_base_name: str,
    *,
    beta_n0: Optional[float] = None,
    mode_n0: Optional[int] = None,
    chi_star_disc: Optional[float] = None,
    meshsize: Optional[int] = None,
    u_pred_plus: Optional[np.ndarray] = None,
    u_pred_minus: Optional[np.ndarray] = None,
    u_pred_amplitude: Optional[float] = None,
    u_pred_is_stable: Optional[bool] = None,
    show_u_pred_amplitude_in_title: bool = True,
) -> None:
    if t_values.size == 0:
        return
    t_end = float(t_values[-1])
    fractions = np.linspace(0.0, 1.0, 6)
    targets = t_end * fractions
    indices: List[int] = []
    for t_target in targets:
        idx = int(np.argmin(np.abs(t_values - t_target)))
        indices.append(idx)

    u_min = float(np.min(u_num))
    u_max = float(np.max(u_num))
    if u_pred_plus is not None and u_pred_minus is not None:
        u_min = float(min(u_min, float(np.min(u_pred_plus)), float(np.min(u_pred_minus))))
        u_max = float(max(u_max, float(np.max(u_pred_plus)), float(np.max(u_pred_minus))))
    u_pad = 0.05 * max(1e-12, u_max - u_min)

    cmap = plt.get_cmap("viridis")
    color_positions = np.linspace(0.15, 0.9, len(indices))
    colors = [cmap(float(p)) for p in color_positions]

    fig, ax_u = plt.subplots(1, 1, figsize=(7.5, 4.8), dpi=300)
    for idx, frac, color in zip(indices, fractions, colors):
        t_here = float(t_values[idx])
        pct = int(round(float(frac) * 100))
        if USE_TEX:
            label = rf"{pct}\% ($t={int(t_here)}$)"
        else:
            label = f"{pct}% (t={int(t_here)})"
        ax_u.plot(x_values, u_num[:, idx], color=color, linewidth=1.6, label=label)

    if u_pred_plus is not None and u_pred_minus is not None:
        pred_color = "black" if u_pred_is_stable else "gray"
        pred_style = ":"
        pred_lw = 1.2
        if USE_TEX:
            pred_label = r"$u_{\rm pred}^\pm$"
        else:
            pred_label = "u_pred±"

        ax_u.plot(
            x_values,
            u_pred_plus,
            color=pred_color,
            linestyle=pred_style,
            linewidth=pred_lw,
            label=pred_label,
        )
        ax_u.plot(
            x_values,
            u_pred_minus,
            color=pred_color,
            linestyle=pred_style,
            linewidth=pred_lw,
            label="_nolegend_",
        )

    title_suffix_tex = ""
    title_suffix_text = ""
    if show_u_pred_amplitude_in_title and u_pred_amplitude is not None:
        title_suffix_tex += rf",\ A_{{\rm pred}}={u_pred_amplitude:.3g}"
        title_suffix_text += f", A_pred={u_pred_amplitude:.3g}"
    if chi_star_disc is not None:
        if meshsize is None:
            title_suffix_tex += rf",\ \chi^{{*,{{\rm disc}}}}={chi_star_disc:.4f}"
            title_suffix_text += f", chi_disc={chi_star_disc:.4f}"
        else:
            title_suffix_tex += rf",\ \chi^{{*,{{\rm disc}}}}(N={int(meshsize)})={chi_star_disc:.4f}"
            title_suffix_text += f", chi_disc(N={int(meshsize)})={chi_star_disc:.4f}"
    if beta_n0 is not None:
        if mode_n0 is None:
            title_suffix_tex += rf",\ \beta={beta_n0:.4g}"
            title_suffix_text += f", beta={beta_n0:.4g}"
        else:
            title_suffix_tex += rf",\ \beta_{{{int(mode_n0)}}}={beta_n0:.4g}"
            title_suffix_text += f", beta_{int(mode_n0)}={beta_n0:.4g}"

    if USE_TEX:
        ax_u.axhline(y=uStar, color="red", linestyle="--", linewidth=0.9, label=r"$u^*$")
        ax_u.set_title(
            rf"$\chi_0={chi0:.4f},\ \chi^*={chi_star:.4f}{title_suffix_tex}$",
            fontsize=11,
        )
        ax_u.set_xlabel(r"$x$")
        ax_u.set_ylabel(r"$u(x,t)$")
    else:
        ax_u.axhline(y=uStar, color="red", linestyle="--", linewidth=0.9, label="u*")
        ax_u.set_title(
            f"chi0={chi0:.4f}, chi*={chi_star:.4f}{title_suffix_text}",
            fontsize=11,
        )
        ax_u.set_xlabel("x")
        ax_u.set_ylabel("u(x,t)")

    ax_u.set_ylim(u_min - u_pad, u_max + u_pad)
    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    ax_u.yaxis.set_major_formatter(y_formatter)

    ax_u.legend(loc="best", fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(f"{file_base_name}_summary6.png", bbox_inches="tight")
    fig.savefig(f"{file_base_name}_summary6.jpeg", bbox_inches="tight")
    plt.close(fig)


def create_static_plots(
    t_mesh: np.ndarray,
    x_mesh: np.ndarray,
    u_data: np.ndarray,
    v_data: np.ndarray,
    uStar: float,
    vStar: float,
    SetupDes: str,
    FileBaseName: str,
) -> None:
    fig_3d = plt.figure(figsize=(15, 6), dpi=300)

    ax_3d_u = fig_3d.add_subplot(121, projection="3d")
    T_grid, X_grid = np.meshgrid(t_mesh, x_mesh, indexing="xy")
    ax_3d_u.plot_surface(T_grid, X_grid, u_data, cmap="viridis", alpha=0.8)

    plt.subplots_adjust(wspace=-0.7)

    U_grid = np.full_like(T_grid, uStar)
    Zero_grid = np.full_like(T_grid, 0)

    ax_3d_u.plot_surface(
        T_grid,
        X_grid,
        U_grid,
        alpha=0.5,
        rstride=100,
        cstride=100,
        color="r",
    )

    ax_3d_u.plot_surface(
        T_grid,
        X_grid,
        Zero_grid,
        alpha=0.2,
        rstride=100,
        cstride=100,
        color="lightgray",
    )

    ax_3d_u.set_xlabel(r"Time $t$")
    ax_3d_u.set_ylabel(r"Space $x$")
    u_min, u_max = u_data.min(), u_data.max()
    ax_3d_u.set_zlim(u_min, u_max)
    ax_3d_u.set_zticks(np.linspace(u_min, u_max, 5))
    ax_3d_u.zaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax_3d_u.set_title("Solution u(t,x)", pad=10)

    ax_3d_v = fig_3d.add_subplot(122, projection="3d")
    ax_3d_v.plot_surface(T_grid, X_grid, v_data, cmap="viridis", alpha=0.8)

    V_grid = np.full_like(T_grid, vStar)

    ax_3d_v.plot_surface(
        T_grid,
        X_grid,
        V_grid,
        alpha=0.5,
        rstride=100,
        cstride=100,
        color="r",
    )

    ax_3d_v.plot_surface(
        T_grid,
        X_grid,
        Zero_grid,
        alpha=0.2,
        rstride=100,
        cstride=100,
        color="lightgray",
    )
    ax_3d_v.set_xlabel(r"Time $t$")
    ax_3d_v.set_ylabel(r"Space $x$")
    v_min, v_max = v_data.min(), v_data.max()
    ax_3d_v.set_zlim(v_min, v_max)
    ax_3d_v.set_zticks(np.linspace(v_min, v_max, 5))
    ax_3d_v.zaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax_3d_v.set_title("Solution v(t,x)", pad=10)

    fig_3d.suptitle(SetupDes, fontsize=10)
    plt.tight_layout()

    fig_3d.savefig(f"{FileBaseName}.png", bbox_inches="tight")
    fig_3d.savefig(f"{FileBaseName}.jpeg", bbox_inches="tight")
    print(
        f"""
    Output files saved:
    - Image: {FileBaseName}.png
    - Image: {FileBaseName}.jpeg
    """
    )
