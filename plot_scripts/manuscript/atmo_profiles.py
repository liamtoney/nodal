import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Geod

from utils import NODAL_WORKING_DIR, get_shots

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

M_PER_KM = 1000  # [m/km]

YLIM = (-0.5, 5)  # [km] Relative to shot z-position

# Parameters for computing sound speed
gamma = 1.4
R = 8.31446  # [J/(mol*K)]
M_air = 0.02896  # [kg/mol]

# Make plot
fig, axs = plt.subplots(
    nrows=2, ncols=2, sharex='col', sharey=True, figsize=(3.47, 5.5)
)
for ax_row, shot_name in zip(axs, ['Y5', 'X5']):

    if shot_name == 'Y5':
        Z_SRC = 734  # [m]
        PROFILE_END = (46.224, -122.031)
    elif shot_name == 'X5':
        Z_SRC = 464  # [m]
        PROFILE_END = (46.138, -122.297)
    else:
        raise ValueError

    shot = get_shots().loc[shot_name]

    # Atmospheric profile from .met file for this run (z is in model z-space!)
    z, t, u, v, _, _ = np.loadtxt(
        NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / f'imush_{shot_name.lower()}.met'
    ).T

    # Adjust z to be relative to shot elevation
    z -= Z_SRC / M_PER_KM

    # Clip to plotting extent
    mask = (z >= YLIM[0]) & (z <= YLIM[1])
    z_plot = np.append(z[mask], YLIM[1])
    t_plot = np.append(t[mask], np.interp(YLIM[1], z, t))
    u_plot = np.append(u[mask], np.interp(YLIM[1], z, u))
    v_plot = np.append(v[mask], np.interp(YLIM[1], z, v))

    # Compute static [dry air] sound speed, see first equation for c_air in
    # https://en.wikipedia.org/wiki/Speed_of_sound#Speed_of_sound_in_ideal_gases_and_air
    c_plot = np.sqrt(gamma * (R / M_air) * t_plot)  # [m/s]

    # Compute dot product
    profile_start = (shot.lat, shot.lon)
    g = Geod(ellps='WGS84')
    az_start_to_end = g.inv(*profile_start[::-1], *PROFILE_END[::-1])[0]
    waz = (90 - np.rad2deg(np.arctan2(v_plot, u_plot))) % 360  # [° from N]
    wmag = np.sqrt(u_plot**2 + v_plot**2)
    angle_diff = waz - az_start_to_end  # Sign does not matter!
    dot_product = wmag * np.cos(np.deg2rad(angle_diff))  # Treating baz as unit vector

    # Make c_eff and find surface value
    c_eff_plot = c_plot + dot_product
    c_eff_surface = np.interp(0, z_plot, c_eff_plot)  # Linear interpolation

    ax1, ax2 = ax_row
    plot_kw = dict(lw=1.5, clip_on=False, solid_capstyle='round', dash_capstyle='round')
    ax1.plot(u_plot, z_plot, color='tab:red', label='Eastward', **plot_kw)
    ax1.plot(v_plot, z_plot, color='tab:blue', label='Northward', **plot_kw)
    ax2.plot(c_plot, z_plot, color='black', alpha=0.4, label='$c$', **plot_kw)
    ax2.plot(c_eff_plot, z_plot, color='black', label='$c_\mathrm{eff}$', **plot_kw)
    ax2.axvline(
        c_eff_surface,
        color='black',
        linestyle=':',
        label='$c_\mathrm{eff}$ (surface)',
        **plot_kw,
    )
    ax1.set_ylabel(f'Elevation relative to shot {shot_name} (km)')
    ax1.set_xlim(-15, 15)
    ax2.set_xlim(300, 350)
    ax1.set_ylim(YLIM)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(10))
    for ax in ax1, ax2:
        xlim = ax.get_xlim()
        ax.fill_between(xlim, -M_PER_KM, 0, color='silver', zorder=2, lw=0.5)
        ax.set_xlim(xlim)
axs[1, 0].set_xlabel('Wind speed (m/s)')
axs[1, 1].set_xlabel('Sound speed (m/s)')
axs[0, 0].legend(
    frameon=False,
    loc='lower left',
    bbox_to_anchor=(0, 1),
    title=r'${\bf Wind\ component}$',
    numpoints=2,
)
axs[0, 1].legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1), numpoints=2)
axs[0, 1].tick_params(left=False, which='both')
axs[1, 1].tick_params(left=False, which='both')
for side in 'top', 'right':
    for ax in axs.flatten():
        ax.spines[side].set_visible(False)
axs[0, 1].spines['left'].set_visible(False)
axs[1, 1].spines['left'].set_visible(False)
for ax in axs.flatten():
    ax.patch.set_alpha(0)

fig.tight_layout(pad=0.2, rect=(0, 0, 1.03, 1.01))
fig.subplots_adjust(hspace=0.1)

# Add grids
grid_kw = dict(
    linestyle=':',
    zorder=-1,
    color='lightgray',
    linewidth=plt.rcParams['grid.linewidth'],
)
for ax in axs.flatten():
    xticks_all = sorted(
        ax.get_xticks().tolist() + ax.xaxis.get_minorticklocs().tolist()
    )
    for loc in xticks_all[2:-2]:
        ax.axvline(loc, **grid_kw)
gs = axs[0, 0].get_gridspec()
for row in 0, 1:
    ax_grid = fig.add_subplot(gs[row, :], sharey=axs[row, 0], zorder=-5)
    for spine in ax_grid.spines.values():
        spine.set_visible(False)
    ax_grid.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    yticks_all = sorted(
        ax_grid.get_yticks().tolist() + ax_grid.yaxis.get_minorticklocs().tolist()
    )
    for loc in yticks_all[3:-2]:
        ax_grid.axhline(loc, **grid_kw)

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'atmo_profiles.png', dpi=600)
