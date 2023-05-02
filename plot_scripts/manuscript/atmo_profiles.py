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

# Parameters for computing sound speed
gamma = 1.4
R = 8.31446  # [J/(mol*K)]
M_air = 0.02896  # [kg/mol]

# Make plot
lw = 1.5
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

    # Compute static [dry air] sound speed, see first equation for c_air in
    # https://en.wikipedia.org/wiki/Speed_of_sound#Speed_of_sound_in_ideal_gases_and_air
    c = np.sqrt(gamma * (R / M_air) * t)  # [m/s]

    # Compute dot product
    profile_start = (shot.lat, shot.lon)
    g = Geod(ellps='WGS84')
    az_start_to_end = g.inv(*profile_start[::-1], *PROFILE_END[::-1])[0]
    waz = (90 - np.rad2deg(np.arctan2(v, u))) % 360  # [° from N]
    wmag = np.sqrt(u**2 + v**2)
    angle_diff = waz - az_start_to_end  # Sign does not matter!
    dot_product = wmag * np.cos(np.deg2rad(angle_diff))  # Treating baz as unit vector

    # Make c_eff and find surface value
    c_eff = c + dot_product
    c_eff_surface = np.interp(0, z, c_eff)  # Linear interpolation
    c_eff_ratio = c_eff / c_eff_surface

    ax1, ax2 = ax_row
    ax1.plot(u, z, lw=lw, color='tab:red', label='Eastward')
    ax1.plot(v, z, lw=lw, color='tab:blue', label='Northward')
    ax2.plot(c, z, lw=lw, color='tab:Gray', label='$c$')
    ax2.plot(c_eff, z, lw=lw, color='black', label='$c_\mathrm{eff}$')
    ax2.axvline(
        c_eff_surface,
        lw=lw,
        color='black',
        linestyle=':',
        label='$c_\mathrm{eff}$ (surface)',
    )
    ax1.set_ylabel(f'Elevation relative to shot {shot_name} (km)')
    ax1.set_xlim(-15, 15)
    ax2.set_xlim(300, 350)
    ax1.set_ylim(-0.5, 5)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(10))
    for ax in ax1, ax2:
        ax.grid(linestyle=':', which='both', color='lightgray', zorder=-1)
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
)
axs[0, 1].legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1))
axs[0, 1].tick_params(left=False, which='both')
axs[1, 1].tick_params(left=False, which='both')
for side in 'top', 'right':
    for ax in axs.flatten():
        ax.spines[side].set_visible(False)
axs[0, 1].spines['left'].set_visible(False)
axs[1, 1].spines['left'].set_visible(False)

fig.tight_layout(pad=0.2)
fig.subplots_adjust(hspace=0.07)

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'atmo_profiles.png', dpi=600)
