import os
import argparse
import sqlite3

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from imscreenpy.visualization import single_cell_vis as sc_vis
from imscreenpy.db_df_processing.db_functions import get_image_df

from imscreenpy.config import Config
from imscreenpy.qc.qc_config import QCconfig
from imscreenpy.qc import qc_functions
from imscreenpy.db_df_processing import db_functions


def get_patches_for_vector(input_df, qc_vector, cell_images_folder, n_patches_to_show=10):
    vsr = sc_vis.Visualizer(input_df, cell_images_folder, x_id='nuclei_Location_Center_X', y_id='nuclei_Location_Center_Y')
    ## sample n_patches_to_show random indices from the qc_vector
    indices = np.random.choice(np.where(qc_vector)[0], size=min(n_patches_to_show, np.sum(qc_vector)), replace=False)
    patches = []
    for index in indices:
        patch = vsr.visualize_single_cell(index, def_size=50, return_xy=False)
        patches.append(patch)
    return patches

def get_patches_and_property_names(input_df, qc_cfg, cell_images_folder):
    patches_per_property = []
    property_names = []
    if not (qc_cfg.max_size_threshold is None):
        qc_vector = ~(input_df[qc_cfg.size_col].to_numpy() < qc_cfg.max_size_threshold)
        if np.sum(qc_vector) > 0:
            patches = get_patches_for_vector(input_df, qc_vector, cell_images_folder)
            patches_per_property.append(patches)
            property_names.append(f"Nucleus size \n >= Max size {qc_cfg.max_size_threshold}")
    if not (qc_cfg.min_size_threshold is None):
        qc_vector = ~(input_df[qc_cfg.size_col].to_numpy() > qc_cfg.min_size_threshold)
        if np.sum(qc_vector) > 0:
            patches = get_patches_for_vector(input_df, qc_vector, cell_images_folder)
            patches_per_property.append(patches)
            property_names.append(f"Nucleus size \n <= Min size {qc_cfg.min_size_threshold}")
    if not (qc_cfg.min_dapi_intensity_threshold is None):
        qc_vector = ~(input_df[qc_cfg.dapi_intensity_col].to_numpy() > qc_cfg.min_dapi_intensity_threshold)
        if np.sum(qc_vector) > 0:
            patches = get_patches_for_vector(input_df, qc_vector, cell_images_folder)
            patches_per_property.append(patches)
            property_names.append(f"DAPI intensity \n <= Min intensity{qc_cfg.min_dapi_intensity_threshold}")
    if not (qc_cfg.min_solidity_threshold is None):
        qc_vector = ~(input_df[qc_cfg.solidity_col].to_numpy() > qc_cfg.min_solidity_threshold)
        if np.sum(qc_vector) > 0:
            patches = get_patches_for_vector(input_df, qc_vector, cell_images_folder)
            patches_per_property.append(patches)
            property_names.append(f"Solidity \n <= Min solidity {qc_cfg.min_solidity_threshold}")
    if not (qc_cfg.max_eccentricity_threshold is None):
        qc_vector = ~(input_df[qc_cfg.eccentricity_col].to_numpy() < qc_cfg.max_eccentricity_threshold)
        if np.sum(qc_vector) > 0:
            patches = get_patches_for_vector(input_df, qc_vector, cell_images_folder)
            patches_per_property.append(patches)
            property_names.append(f"Eccentricity \n >= Max eccentricity {qc_cfg.max_eccentricity_threshold}")
    if not (qc_cfg.min_entropy_threshold is None):
        qc_vector = ~(input_df[qc_cfg.texture_entropy_col].to_numpy() > qc_cfg.min_entropy_threshold)
        if np.sum(qc_vector) > 0:
            patches = get_patches_for_vector(input_df, qc_vector, cell_images_folder)
            patches_per_property.append(patches)
            property_names.append(f"Entropy \n <= Min entropy {qc_cfg.min_entropy_threshold}")
    if not (qc_cfg.min_contrast_threshold is None):
        qc_vector = ~(input_df[qc_cfg.texture_contrast_col].to_numpy() > qc_cfg.min_contrast_threshold)
        if np.sum(qc_vector) > 0:
            patches = get_patches_for_vector(input_df, qc_vector, cell_images_folder)
            patches_per_property.append(patches)
            property_names.append(f"Contrast \n <= Min contrast {qc_cfg.min_contrast_threshold}")
    if not (qc_cfg.texture_sum_min_value is None):
        qc_vector = ~(input_df[qc_cfg.texture_sum_col].to_numpy() > qc_cfg.texture_sum_min_value)
        if np.sum(qc_vector) > 0:
            patches = get_patches_for_vector(input_df, qc_vector, cell_images_folder)
            patches_per_property.append(patches)
            property_names.append(f"Texture sum \n <= Min value {qc_cfg.texture_sum_min_value}")
    if not (qc_cfg.info_meas2_exclude_value is None):
        qc_vector = ~(input_df[qc_cfg.texture_info_meas2_col].to_numpy() != qc_cfg.info_meas2_exclude_value)
        if np.sum(qc_vector) > 0:
            patches = get_patches_for_vector(input_df, qc_vector, cell_images_folder)
            patches_per_property.append(patches)
            property_names.append(f"Info meas2 \n = {qc_cfg.info_meas2_exclude_value}")
    if not (qc_cfg.correlation_exclude_values is None):
        qc_vector = np.isin(input_df[qc_cfg.correlation_col].to_numpy(), qc_cfg.correlation_exclude_values)
        if np.sum(qc_vector) > 0:
            patches = get_patches_for_vector(input_df, qc_vector, cell_images_folder)
            patches_per_property.append(patches)
            property_names.append(f"Correlation in {qc_cfg.correlation_exclude_values}")
    return patches_per_property, property_names


def plot_patches_per_property(patches, property_names, savepath, n_patches_to_show=10):
    n_properties = len(patches)
    fig, ax = plt.subplots(figsize=(n_properties*2.5, n_patches_to_show*2.5), nrows=n_patches_to_show, ncols=n_properties)
    for col_index, property_name in enumerate(property_names):
        for row_index in range(n_patches_to_show):
            if row_index < len(patches[col_index]):
                ax[row_index, col_index].imshow(patches[col_index][row_index])
            else:
                ax[row_index, col_index].axis('off')
            if row_index == 0:
                ax[row_index, col_index].set_title(property_name)
            ax[row_index, col_index].axis('off')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)    
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare QC visualization patches")
    parser.add_argument("db_path", type=str, help="Path to the input database")
    parser.add_argument("savepath", type=str, help="Path to save the visualization patches")
    parser.add_argument("cell_images_folder", type=str, help="Path to the folder containing cell images")
    parser.add_argument("--cfg_folder_path", type=str, help="Path to the QC configuration file")
    parser.add_argument('--target_fluorophores', type=str, nargs='+', default=['GFP', 'PE', 'DNA'], help="List of target fluorophores to use for the analysis")
    parser.add_argument("--n_patches_to_show", type=int, default=10, help="Number of patches to show per property")

    args = parser.parse_args()

    target_connection = sqlite3.connect(args.db_path)



    target_fluorophores = args.target_fluorophores
    ## set up default config
    cfg = Config(None)
    cfg.db_im_table_name = 'Per_Image'
    cfg.db_cell_table_name = 'Per_Object'
    cfg.set_intensity_columns(target_fluorophores = target_fluorophores, intensity_column_suffix='')

    print('Set up config')

    
    image_df = get_image_df(target_connection, target_fluorophores, cfg=None, other_qc_columns=['Image_Count_nuclei'], \
                  intensity_column_prefix='Image_Intensity_MeanIntensity_', metadata_well_column='Image_Metadata_Well',\
                    metadata_column_column='Image_Metadata_column',\
                        metadata_field_column='Image_Metadata_field', metadata_row_column='Image_Metadata_row', db_im_table_name='Per_Image')
    im_outlier_df = qc_functions.get_image_df_outlier_wells(image_df, target_fluorophores, qc_plot_savename=None, plot_qc=False, be_relaxed=True, ignore_cellnumbers=True)
    
    print('Prepared image dataframe')

    exclusion_column_patterns=[['index'], ['nuclei', 'APC'], ['nuclei', 'PE'], ['nuclei', 'GFP'], ['cells_dist', 'Location'],\
         ['cells_dist', 'Moment'], ['cells_dist', 'Tensor'], ['cells_dist', 'Displacement'], ['cells_dist', 'Quartile'], ['cells_dist', 'Integrated']]
    target_column_list = cfg.get_all_set_columns()
    
    full_object_df = db_functions.full_df_from_conn(target_connection, cfg.db_cell_table_name, exclusion_wells=im_outlier_df['Image_Metadata_Well'].unique(), exclusion_column_patterns=exclusion_column_patterns, column_subset=target_column_list)
    print('Got full dataframe with shape {}'.format(full_object_df.shape))
    #cfg = Config(args.cfg_folder_path, cfg_predix='aml_')
    qc_cfg = QCconfig(args.cfg_folder_path, cfg_prefix='aml_')
    ## get patches for each qc exclusion property
    patches_per_property, property_names = get_patches_and_property_names(full_object_df, qc_cfg, args.cell_images_folder)
    ## we also load patches with cells that passed qc to have a comparison within the same figure
    qc_passed_vector = qc_functions.make_morphology_qc_vector(full_object_df, qc_cfg)
    qc_passed_patches = get_patches_for_vector(full_object_df, qc_passed_vector, args.cell_images_folder, n_patches_to_show=10)
    all_patches = [qc_passed_patches] + patches_per_property
    all_property_names = ['QC passed'] + property_names
    print('Got patches for the followig properties {}'.format(property_names))
    plot_patches_per_property(all_patches, all_property_names, args.savepath)
