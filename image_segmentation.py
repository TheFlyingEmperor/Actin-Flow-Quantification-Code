# -*- coding: utf-8 -*-
"""
Focal Adhesion Segmentation Module

Created on Tue Jan  7 09:15:10 2025
@author: Owner

Functions for segmenting and quantifying focal adhesions using Cellpose.
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from skimage.exposure import equalize_adapthist
from scipy.ndimage import gaussian_filter
import pyclesperanto as cle


def quantify_frame(label, img, um_per_pix,):
    """
    Quantify properties of segmented regions in a single frame.
    
    Parameters
    ----------
    label : ndarray
        Label image from segmentation.
    img : ndarray
        Paxillin intensity image.
    Returns
    -------
    data : dataframe
        DataFrame with region properties.
    """
    data = []
    region_list = measure.regionprops(label, img)
    
    for region in region_list:
        if region.axis_minor_length != 0:
            
            data.append({
                'pax_int': region.intensity_mean,
                'area': region.area,
                'length': region.axis_major_length*um_per_pix,
                'width': region.axis_minor_length*um_per_pix,
                'aspect_ratio': region.axis_major_length/region.axis_minor_length
            })
            
    return pd.DataFrame(data)

def segment_frame(image, model, smooth_kernel, clip_limit, top_hat,
                  flow_threshold=0.4, cellprob_threshold=-0.5, min_area=0, min_solidity = 0,
                  plot=False):
    """
    Segment focal adhesions in a single frame using Cellpose.
    
    Parameters
    ----------
    image : ndarray
        Input image to segment.
    model : CellposeModel
        Pre-trained Cellpose model.
    smooth_kernel : float
        Standard deviation for Gaussian smoothing.
    clip_limit : float
        Clip limit for CLAHE enhancement.
    top_hat : bool
        Whether to apply top-hat background removal.
    flow_threshold : float, optional
        Cellpose flow threshold parameter.
    cellprob_threshold : float, optional
        Cellpose cell probability threshold.
    min_area : int, optional
        Minimum area in pixels for a label to be kept.
    plot : bool, optional
        Whether to display segmentation overlay.
        
    Returns
    -------
    labels : ndarray
        Segmentation label image with small regions removed.
    """
    # Normalize image
    maximum = np.max(image)
    image = image / maximum
    
    # Optional top-hat background removal
    if top_hat:
        image = np.asarray(cle.top_hat_box(image, image, 30, 30))
    
    # Enhance contrast and smooth
    image = equalize_adapthist(image, clip_limit=clip_limit)
    image = gaussian_filter(image, smooth_kernel)
    
    # Run Cellpose segmentation
    labels = model.eval(
        x=image,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=True
    )[0]
    
    # Filter out labels smaller than min_area
    if min_area > 0:
        # Get properties of each labeled region
        regions = measure.regionprops(labels)
        
        # Create a mask of labels to remove
        for region in regions:
            if region.area < min_area:
                labels[labels == region.label] = 0
            
            if region.solidity < min_solidity:
                labels[labels == region.label] = 0
        
        # Relabel to ensure consecutive label numbers
        labels = measure.label(labels > 0)
    
    # Optional visualization
    if plot:
        plt.imshow(image, cmap='gray')
        plt.imshow(
            np.ma.masked_where(labels == 0, labels),
            cmap='jet',
            alpha=0.5
        )
        plt.show()
    
    return labels

#%%

from skimage.io import imread
from cellpose import models
import pandas as pd
import os


# File paths
FOLDER = '/media/rico/Seagate Portable Drive/Research/Splitting_FA_project/milda_images/MEF_mNG-Paxillin_mScar-LifeAct_Glass_2'
um_per_px = 0.0313 #microns per pixel
target_filename = 'AVG_paxillin_SIM.tif'
width_list = []
all_data_list = []
min_area = 400 #min area in pixels required for FA to be quantified
min_solidity = 0.75 #min fraction of label that is covered by convex hull to be considered a FA; 
                    #acts as heuristic measurement of FA irregularity (irregular object -> not FA)

counter = 0
for root, dirs, files in os.walk(FOLDER):
        # Check if target file is in current directory
        if target_filename in files:
            counter += 1
            full_path = os.path.join(root, target_filename)
            print(f'Found Image: {full_path}')
            image = imread(full_path)
            
            model = models.CellposeModel(gpu=True, pretrained_model='cpsam_20260204_112033')
            labels = segment_frame(image, model, 0.5, 0.01, False, plot=True, min_area = min_area, min_solidity = min_solidity)
            dataframe = quantify_frame(labels, image, um_per_px)
            width_list.append(dataframe['width'])
            all_data_list.append(dataframe)

result_df = pd.concat(width_list, ignore_index=True)
result_df.to_excel(FOLDER + '/compiled_width_automatic.xlsx')            
ar_df = pd.concat(all_data_list, ignore_index = True)
ar_df.to_excel(FOLDER + '/all_data_compiled_auto.xlsx')