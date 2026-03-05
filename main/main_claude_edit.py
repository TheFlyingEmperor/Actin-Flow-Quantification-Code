# -*- coding: utf-8 -*-
"""
Actin Flow Quantification Pipeline (Flow Only)

This script performs:
1. Actin flow quantification using PIV (Particle Image Velocimetry)
2. Flow visualization overlaid on paxillin channel
"""

# %% Imports
import sys
sys.path.insert(0, '../utilities')

from skimage.io import imread, imsave
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

import util_flow_only as util

# %% Configuration

# File paths
FOLDER = '/media/rico/9d9eec64-93bc-4c31-9305-8348e35db5a4/Research/Flux_Dependent_FA_Disassembly/actin_flux_data/raw_data/ActinFlow_June24_2025/cells_w_retracting_end'
IMAGE_NAME = 'Cell_03'

# Flow quantification settings
FLOW_CONFIG = {
    'winsize': 30,
    'ensemble_winsize': 11,      # Odd windows are ideal for proper centering
    'overlap': 15,              # 3 * winsize // 5
    'time_btwn_frames': 10,     # seconds between frames
    'nm_per_pix': 73,           # Width of a pixel in nm
    'error_thresh': 1.1,        # Error threshold for noisy correlation peaks
}

# %% Load and Prepare Data

# Load multichannel movie
movie = imread(f'{FOLDER}/{IMAGE_NAME}.tif')

# Split channels
imstack = movie[:, :, :, 0]         # Channel 0: Actin speckles
pax_imstack = movie[:, :, :, 1]     # Channel 1: Paxillin
actin_imstack = movie[:, :, :, 2]   # Channel 2: Actin

del movie  # Free memory

# %% Initialize Data Structures

# Create ensemble windows for PIV
a, b = util.create_ensemble_windows(
    FLOW_CONFIG['ensemble_winsize'], 
    imstack.shape[0]
)

# Initialize speed collector
speed_list = []

# Normalization for paxillin visualization
norm = colors.Normalize(vmin=np.min(pax_imstack), vmax=np.max(pax_imstack))

# %% Process Each Ensemble Window

for i, j in zip(a, b):
    # Compute ensemble-averaged correlation between frames of actin speckles
    mean_corr = util.ensemble_average_corr(
        imstack[i:j].copy(),
        winsize=FLOW_CONFIG['winsize'],
        overlap=FLOW_CONFIG['overlap']
    )
    
    # Compute flow field
    x, y, u, v = util.compute_flow_field(
        mean_corr,
        im=imstack[(i + j) // 2].copy(),
        winsize=FLOW_CONFIG['winsize'],
        overlap=FLOW_CONFIG['overlap'],
        error_thresh=FLOW_CONFIG['error_thresh']
    )
    
    # Get center frame index
    center_frame = (i + j) // 2
    
    # Plot flow field overlaid on paxillin
    speed = util.plot_flow_on_paxillin(
        x=x, y=y, u=u, v=v,
        im=imstack[center_frame].copy(),
        pax_im=pax_imstack[center_frame].copy(),
        actin=actin_imstack[center_frame].copy(),
        frame=center_frame,
        nm_per_pix=FLOW_CONFIG['nm_per_pix'],
        time_btwn_frames=FLOW_CONFIG['time_btwn_frames'],
        norm=norm
    )
    
    # Accumulate results
    speed_list.append(speed)

# %% Save Results

speed_list = np.array(speed_list)

# Save velocity images
imsave(f'{FOLDER}/{IMAGE_NAME}Flux_Vel.tif', speed_list.astype(np.float32))

print(f"Saved flow velocity map to {FOLDER}/{IMAGE_NAME}Flux_Vel.tif")